from langchain_chroma import Chroma

import os
import yaml
import time, datetime
import argparse
import asyncio
from dotenv import load_dotenv

import logging
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.load_config import cfgs
from rag_info_extractor.utils.embedder import HFEmbedder
from rag_info_extractor.utils.llm_connector import OllamaLLM
from rag_info_extractor.rag_pipeline import RAGPipeline
from rag_info_extractor.extract_info import (
    extract_and_save_all_info,
    aextract_and_save_all_info,
)
from rag_info_extractor.info_schema.utils import load_classes_from_path


logger = logging.getLogger(__name__)


def main():
    # 1. Setup CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )  # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    parser.add_argument("--llm-model", type=str, help="LLM Model for RAG pipeline")
    parser.add_argument("--extractor-llm", type=str, help="LLM Model for RAG pipeline")
    parser.add_argument(
        "--chunks-type",
        type=str,
        choices=[
            "custom_chunks",
            "fixed_size_chunks",
            "semantic_chunks",
            "custom_chunks_2",
        ],
        help="Chunking method used to extract context from",
    )
    args = parser.parse_args()

    # 2. Load config Vars
    LLM_MODEL = cfgs.get("LLM_MODEL")
    EXTRACTOR_LLM = cfgs.get("EXTRACTOR_LLM")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    RUN_ASYNC = cfgs.get("RUN_ASYNC", False)
    USE_GOOGLE_API = cfgs.get("USE_GOOGLE_API")

    # 3. Load env vars
    load_dotenv("../.env")

    # 4. Modify config setting from CLI for automated testing if needed
    if args.llm_model:
        LLM_MODEL = args.llm_model
        cfgs["LLM_MODEL"] = LLM_MODEL
    if args.extractor_llm:
        EXTRACTOR_LLM = args.extractor_llm
        cfgs["EXTRACTOR_LLM"] = LLM_MODEL
    if args.chunks_type:
        CHUNKS_TYPE = args.chunks_type
        cfgs["CHUNKS_TYPE"] = CHUNKS_TYPE

    # 5. Configure logging settings
    RUN_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    LOGDIR = os.path.join(BASE_DIR, "logs", "extract_info_py")
    os.makedirs(LOGDIR, exist_ok=True)
    configure_logging(
        default_level=logging.DEBUG if args.verbose else logging.INFO,
        logfile=os.path.join(LOGDIR, f"{RUN_TIME}.log"),
    )

    # 6. Create output dir for saving outputs and settings of the run
    OUTPUT_SAVE_DIR = os.path.join(
        BASE_DIR, "runs", DATASET_TYPE, CHUNKS_TYPE, f"run_{RUN_TIME}"
    )
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
    logger.info("Output will be saved to %s", OUTPUT_SAVE_DIR)
    # Save config
    with open(os.path.join(OUTPUT_SAVE_DIR, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfgs, f)

    # 7. Load Doc store (for parent/large chunks) and Vector store (for children/small chunks)
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(
        BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE
    )
    VECTOR_STORE_PATH = os.path.join(
        BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE
    )

    if not os.path.exists(DOC_STORE_LARGE_CHUNKS_PATH) or not os.scandir(
        DOC_STORE_LARGE_CHUNKS_PATH
    ):
        logger.exception(
            "Large Chunks Path: %s  does not exist", DOC_STORE_LARGE_CHUNKS_PATH
        )
        raise FileNotFoundError("Large Chunks Path does not exist")
    if not os.path.exists(VECTOR_STORE_PATH) or not os.scandir(VECTOR_STORE_PATH):
        logger.exception("Vector DB Path: %s  does not exist", VECTOR_STORE_PATH)
        raise FileNotFoundError("Vector DB Path does not exist")

    embedding = HFEmbedder(normalize_embeddings=True)
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=VECTOR_STORE_PATH,
        collection_name="pdf_chunks",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 8}
    )

    # 8. Get all azienda names in vector/doc store
    nome_delle_aziende = set(
        (
            vector_store.get()["metadatas"][i].get("azienda"),
            vector_store.get()["metadatas"][i].get("filename"),
        )
        for i in range(len(vector_store.get()["ids"]))
    )
    nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    nome_delle_aziende = [x for x in nome_delle_aziende if x[0] != "unicredit s.p.a."]
    azienda_name_records = [
        x[0] for x in nome_delle_aziende if x[0] != "unicredit s.p.a."
    ]
    # =========================== DEFINE nome_aziende MANUALLY  ===========================
    # nome_delle_aziende = [
    #     ('compagnie de participation hotelliere et touristique', '8049135570002.pdf')
    # ]
    # azienda_name_records = [
    #     'compagnie de participation hotelliere et touristique',
    # ]
    # ============================================ XXX ============================================

    # 9. Load the schemas of info to be etracted
    classes = load_classes_from_path(
        os.path.join(BASE_DIR, "src", "rag_info_extractor", "info_schema", "schemas")
    )

    # 10. Load the RAG pipeline
    rag_obj = RAGPipeline(
        db_retriever=retriever,
        azienda_name_records=azienda_name_records,
        llm_model=LLM_MODEL,
        doc_store_path=DOC_STORE_LARGE_CHUNKS_PATH,
        pages_joining_str=PAGES_JOINING_STR,
        use_google_api=USE_GOOGLE_API,
        run_async=RUN_ASYNC,
    )
    logger.info("Initialized RAG Pipeline.")
    rag_obj.save_DAG_diagram(OUTPUT_SAVE_DIR)
    logger.info("The DAG for RAG Pipeline saved to %s", OUTPUT_SAVE_DIR)

    # 11. Define LLM for extractions
    llm_for_extraction = OllamaLLM(llm_model=EXTRACTOR_LLM, temperature=0)

    # Extract info
    if RUN_ASYNC:
        logger.info("Extracting info asynchronously.")
        asyncio.run(
            aextract_and_save_all_info(
                rag_pipeline=rag_obj,
                nome_delle_aziende=nome_delle_aziende,
                llm_json=llm_for_extraction,
                save_dir=OUTPUT_SAVE_DIR,
                use_google_api=USE_GOOGLE_API,
            )
        )
    else:
        logger.info("Extracting info synchronously.")
        extract_and_save_all_info(
            rag_pipeline=rag_obj,
            nome_delle_aziende=nome_delle_aziende,
            llm_json=llm_for_extraction,
            save_dir=OUTPUT_SAVE_DIR,
            use_google_api=USE_GOOGLE_API,
        )
    logger.info("Completed Extraction. Outputs saved to %s", OUTPUT_SAVE_DIR)


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )
