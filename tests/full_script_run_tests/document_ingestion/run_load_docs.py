import os 
import time
import argparse
import logging
import asyncio
from pathlib import Path

from rag_info_extractor.document_ingestion.load_docs import aload_pdfs
from dotenv import load_dotenv

from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.load_config import cfgs

logger = logging.getLogger(__name__)

def main():
    # 1. Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()
    configure_logging(
        default_level=logging.DEBUG if args.verbose else logging.INFO
    )

    # 2. CONFIG FILE SETTINGS:
    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    LLM_MODEL = cfgs.get("LLM_MODEL")
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM") 
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    MAX_EMBED_TOKENS = cfgs.get("MAX_EMBED_TOKENS")
    READ_MODE = cfgs.get("READ_MODE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = Path(__file__).resolve().parents[3]

    DATASET_DIR = os.path.join(BASE_DIR, "data", "pdfs", DATASET_TYPE) #f"../data/pdfs/{DATASET_TYPE}"

    # 3. Load env_vars
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    EMBEDDING_MODEL_NAME_ENV = (
        EMBEDDING_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
    )
    EMBEDDING_MODEL_PATH = os.environ.get(
        EMBEDDING_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME
    )

    # 4. Load pdfs
    logger.info(f"Loading the documents: {os.listdir(DATASET_DIR)}")
    output = asyncio.run(aload_pdfs(
        folder = DATASET_DIR,
        HF_embedding_model_name = EMBEDDING_MODEL_PATH,
        evaluator_llm = EVALUATOR_LLM,
        llm_model = LLM_MODEL,
        max_embed_tokens = MAX_EMBED_TOKENS,
        chunks_type = CHUNKS_TYPE,
        read_mode = READ_MODE,
        pages_joining_str = PAGES_JOINING_STR    
    ))

    parent_chunks, children_chunks = output.get("parent_chunks", []), output.get("children_chunks", [])

    # 5.Save output in temp txt file
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("OUTPUT FOR load_docs.py\n\n")
        
        f.write("PARENT CHUNKS: \n\n")
        for c in parent_chunks:
            f.write(f"\n{"-"*50} CHUNK ID: {c.metadata.get("chunk_id")} \t chunking_method: {c.metadata.get("pattern_name")} {"-"*50}\n")
            f.write(f"Azienda Name: {c.metadata.get("azienda")}\n")
            f.write(f"Azienda Sede: {c.metadata.get("sede")}\n")
            f.write(f"{c.page_content}\n\n")

        # f.writelines([f"CHUNK ID = {c.metadata.get("chunk_id")}\n{c.page_content}\n\n" for c in parent_chunks])
        f.write(f"{"x"*100}\n")

        f.write("Children Chunks\n")
        for c in children_chunks:
            f.write(f"\n{"-"*50} CHUNK ID: {c.metadata.get("chunk_id")} {"-"*50}\n")
            f.write(f"{c.page_content}\n\n")



if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )