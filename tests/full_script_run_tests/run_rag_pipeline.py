import argparse
import asyncio
import datetime

# logging relative
import logging
import os
import time
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_info_extractor.rag_pipeline import RAGPipeline
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.embedder import HFEmbedder
from rag_info_extractor.utils.load_config import cfgs
from torch import Use

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

    # 2. CONFIG FILE SETTINGS:
    LLM_MODEL = cfgs.get("LLM_MODEL")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = Path(__file__).resolve().parents[2]

    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(
        BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE
    )
    VECTOR_STORE_PATH = os.path.join(
        BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE
    )
    assert os.path.exists(
        DOC_STORE_LARGE_CHUNKS_PATH
    ), "DOC_STORE_LARGE_CHUNKS_PATH not found"
    assert os.path.exists(VECTOR_STORE_PATH), "VECTOR_STORE_PATH not found"

    # 3. Configure logging settings
    RUN_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    LOG_DIR = os.path.join(BASE_DIR, "logs", "rag_pipeline_py")
    os.makedirs(LOG_DIR, exist_ok=True)
    configure_logging(
        default_level=logging.DEBUG if args.verbose else logging.INFO,
        logfile=os.path.join(LOG_DIR, f"{RUN_TIME}.log"),
    )

    logger.info(
        f'Logging for {"-"*30} rag_information_extractor/scripts/rag_pipeline_components.py'
    )  ###
    logger.info(f"LLM model used: {LLM_MODEL}")

    # 4. Load Vector and Doc store
    embedding = HFEmbedder(normalize_embeddings=True)
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=VECTOR_STORE_PATH,
        collection_name="pdf_chunks",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 8}
    )

    # Get all azienda names in vector/doc store
    nome_delle_aziende = set(
        (
            vector_store.get()["metadatas"][i].get("azienda"),
            vector_store.get()["metadatas"][i].get("filename"),
        )
        for i in range(len(vector_store.get()["ids"]))
    )
    nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    azienda_name_records = [x[0] for x in nome_delle_aziende]

    logger.info("Loaded Vector + Doc Store.")  ###

    # 5. Initialize RAG pipeline
    logger.info("Initializing RAG Pipeline...")  ###
    try:
        rag_obj = RAGPipeline(
            db_retriever=retriever,
            azienda_name_records=azienda_name_records,
            llm_model=LLM_MODEL,
            doc_store_path=DOC_STORE_LARGE_CHUNKS_PATH,
            pages_joining_str=PAGES_JOINING_STR,
            run_async=RUN_ASYNC,
            use_google_api=USE_GOOGLE_API,
        )
        logger.info("Initialized RAG Pipeline.")  ###
    except:
        logger.exception("message")
    else:
        # save the DAG flow for RAG nodes/Pipeline
        rag_obj.save_DAG_diagram("")
        logger.info(f"The DAG for RAG Pipeline saved to {""}")

    # 6.Setup RAG Query/question/experiment
    USER_QUERY = "Agli amministratori spetta il rimborso delle spese? Informazione richiesto per la società: 2KIND SRL"  # Query and Aziende (EXAMPLE)
    # USER_QUERY = "I soci possono assegnare un compenso agli amministratori? In che misura? QUERY: compenso amministratori limiti massimi e criteri di determinazione indennità Nome della società: 2kind srl"
    logger.info("Running query...")  ###
    if RUN_ASYNC:
        ai_response = asyncio.run(rag_obj.aget_response(query=USER_QUERY))
    else:
        ai_response = rag_obj.get_response(query=USER_QUERY)

    # 7. Save outputs to output_temp.txt
    def write_outputs_to_file():
        # Obtain retrieved chunks texts
        retrieved_docs_ids = rag_obj.retrieved_docs_ids
        retrieved_vs_chunk_ids = [
            i
            for i, m in enumerate(vector_store.get().get("metadatas", []))
            if m.get("chunk_id") in retrieved_docs_ids.get("children", [])
        ]
        retrieved_vs_chunks = [
            c
            for i, c in enumerate(vector_store.get().get("documents", []))
            if i in retrieved_vs_chunk_ids
        ]  # vector_store chunks

        retrieved_parents_keys = retrieved_docs_ids.get("parents", [])
        retrieved_ds_chunks = ["" for i in range(len(retrieved_parents_keys))]
        for i, id in enumerate(retrieved_parents_keys):
            with open(
                f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content/{id}", encoding="utf-8"
            ) as f:
                retrieved_ds_chunks[i] = f.read()

        # Obtain re_ranked_chunks
        re_ranked_docs_ids = rag_obj.re_ranked_docs_ids
        re_ranked_vs_chunk_ids = [
            i
            for i, m in enumerate(vector_store.get().get("metadatas", []))
            if m.get("chunk_id") in re_ranked_docs_ids.get("children", [])
        ]
        re_ranked_vs_chunks = [
            c
            for i, c in enumerate(vector_store.get().get("documents", []))
            if i in re_ranked_vs_chunk_ids
        ]  # vector_store chunks

        re_ranked_parents_keys = re_ranked_docs_ids.get("parents", [])
        re_ranked_ds_chunks = ["" for i in range(len(re_ranked_parents_keys))]
        for i, id in enumerate(re_ranked_parents_keys):
            with open(
                f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content/{id}", encoding="utf-8"
            ) as f:
                re_ranked_ds_chunks[i] = f.read()

        # Store contexts/query in output_temp.txt
        with open("output_temp", "w", encoding="utf-8") as f:
            f.write(f"## OUTPUT FOR: rag_pipeline_components.py \n{RUN_TIME}\n")
            f.write(f"Async: {RUN_ASYNC}\n\n")
            f.write(f"USER QUERY: {USER_QUERY}\n\n")
            f.write(f"Optimied Query: {rag_obj.optimized_query}\n\n")
            f.write(f"LLM ANSWER: {ai_response}\n\n")

            f.write(f"\n{"x"*100}\n")

            # Retrieved documents
            f.write("\nRETRIEVER DOCs...\n\n")
            f.write(f"Retrieved doc ids: {retrieved_docs_ids}\n\n")
            f.write(f"VECTOR STORE chunks: \n")
            for i, c in enumerate(retrieved_vs_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")
            f.write(f"{"-"*100}\n{"-"*100}\n")
            f.write(f"DOC STORE chunks: \n")
            for i, c in enumerate(retrieved_ds_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")

            f.write(f"\n{"x"*100}\n")

            # Re-Ranked documents
            f.write("\nRE-RANKED DOCs...\n\n")
            f.write(f"Re-Ranked doc ids: {re_ranked_docs_ids}\n\n")
            f.write(f"VECTOR STORE chunks: \n")
            for i, c in enumerate(re_ranked_vs_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")
            f.write(f"{"-"*100}\n{"-"*100}\n")
            f.write(f"DOC STORE chunks: \n")
            for i, c in enumerate(re_ranked_ds_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")

    logger.info("Saving outputs...")
    write_outputs_to_file()


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})
    RUN_ASYNC = True
    USE_GOOGLE_API = True

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )
