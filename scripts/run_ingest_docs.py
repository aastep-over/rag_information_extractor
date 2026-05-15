import os
import yaml
import time
import argparse
from dotenv import load_dotenv

# logging relative
import logging
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.document_ingestion.ingest_docs import (
    load_docs_from_dir,
    save_parent_chunk_bytes,
    embed_child_chunks,
)
from rag_info_extractor.utils.load_config import cfgs

logger = logging.getLogger(__name__)


def main():

    # 1. Load config Vars
    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    LLM_MODEL = cfgs.get("LLM_MODEL")
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    MAX_EMBED_TOKENS = cfgs.get("MAX_EMBED_TOKENS")
    READ_MODE = cfgs.get("READ_MODE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    DATASET_DIR = os.path.join(BASE_DIR, "data", "pdfs", DATASET_TYPE)
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(
        BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, f"{CHUNKS_TYPE}"
    )
    VECTOR_STORE_PATH = os.path.join(
        BASE_DIR, "data", "vector_dbs", DATASET_TYPE, f"{CHUNKS_TYPE}"
    )

    # 2. Load env_vars
    load_dotenv("../.env")
    EMBEDDING_MODEL_NAME_ENV = (
        EMBEDDING_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
    )
    EMBEDDING_MODEL_PATH = os.environ.get(
        EMBEDDING_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME
    )

    # configure logging
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    configure_logging(
        default_level=logging.DEBUG if args.verbose else logging.INFO,
        logfile=os.path.join(BASE_DIR, "info_extractor.log"),
    )
    logger.info(f'Loading the documents: {", ".join(os.listdir(DATASET_DIR))}')

    # load and split docs in parent & child chunks
    parent_chunks, children_chunks = load_docs_from_dir(
        dataset_dir=DATASET_DIR,
        chunks_type=CHUNKS_TYPE,
        HF_embedding_model_name=EMBEDDING_MODEL_PATH,
        evaluator_llm=EVALUATOR_LLM,
        llm_model=LLM_MODEL,
        max_embed_tokens=MAX_EMBED_TOKENS,
        read_mode=READ_MODE,
        pages_joining_str=PAGES_JOINING_STR,
    )

    logger.info(
        f"Loaded {len(parent_chunks)} Parent chunks from {len(set(d.metadata['filename'] for d in parent_chunks))} PDFs"
    )
    logger.info(
        f"Loaded {len(children_chunks)} Children chunks from {len(set(d.metadata['filename'] for d in children_chunks))} PDFs"
    )

    # save parent chunks and embed child chunks
    if parent_chunks:
        save_parent_chunk_bytes(
            parent_chunks, doc_store_dir=DOC_STORE_LARGE_CHUNKS_PATH
        )
    if children_chunks:
        embed_child_chunks(children_chunks, vector_store_dir=VECTOR_STORE_PATH)


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})

    main()

    logger.info(
        f'Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}'
    )
