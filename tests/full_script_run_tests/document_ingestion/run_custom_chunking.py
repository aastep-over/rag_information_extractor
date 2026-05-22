from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

import fitz
import os
import time
import argparse
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from rag_info_extractor.document_ingestion.custom_chunking import custom_chunking
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
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    MAX_EMBED_TOKENS = cfgs.get("MAX_EMBED_TOKENS")
    READ_MODE = cfgs.get("READ_MODE", "single")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = Path(__file__).resolve().parents[3]
    PDF_LOADER = cfgs.get("PDF_LOADER", "./")

    DATASET_DIR = os.path.join(BASE_DIR, "data", "pdfs", DATASET_TYPE)

    # 3. Load env_vars
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    EMBEDDING_MODEL_NAME_ENV = (
        EMBEDDING_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
    )
    EMBEDDING_MODEL_PATH = os.environ.get(
        EMBEDDING_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME
    )

    # Load pdf
    docs: list[Document] = []
    if PDF_LOADER == "pymupdf":
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info("Document exists: %s", os.path.exists(path))
            doc = fitz.open(path)

            meta = doc.metadata if isinstance(doc.metadata, dict) else {}

            meta = {
                **meta,
                **{
                    "source": Path(doc.name).name if doc.name else None,
                    "total_pages": doc.page_count,
                },
            }
            text = PAGES_JOINING_STR.join(
                page.get_text("text") or "" for page in doc
            )  # type: ignore (return type of get_text is not only str)
            docs.append(Document(page_content=text, metadata=meta))
            doc.close()
    else:
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info("Document exists: %s", os.path.exists(path))
            loader = PyPDFLoader(
                path,
                mode=READ_MODE,
                pages_delimiter=PAGES_JOINING_STR,
            )
            docs.extend(loader.load())

    logger.info("Docs Loaded")
    # Define text splitter and tokenizer
    chunk_size = 430
    chunk_overlap = 105
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, use_fast=True)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    # Run custom_chunking_function
    logger.info("Creating Chunks...")
    chunks = asyncio.run(
        custom_chunking(
            docs,
            text_splitter,
            tokenizer,
            evaluator_llm=EVALUATOR_LLM,
            max_embed_tokens=MAX_EMBED_TOKENS,
            use_llm=True,
            pages_joining_str=PAGES_JOINING_STR,
        )
    )

    logger.info("Chunks created. Saving...")
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("OUTPUT FOR custom_chunking.py\n\n")
        f.write("Whole Articles: \n")

        f.write("PARENT CHUNKS: \n\n")
        parent_chunks = chunks.get("whole_articles")
        for c in parent_chunks:
            f.write(
                f'\n{"-" * 50} CHUNK ID: {c.metadata.get("chunk_id")} {"-" * 50}\n'
            )
            f.write(f"{c.page_content}\n\n")

        f.write(f'{"x" * 100}\n')

        f.write("DOCS NOT SPLIT: \n\n")
        f.write(f'last_parent_id =  {chunks.get("last_parent_id")}\n')
        f.write(f'last_child_id =  {chunks.get("last_child_id")}\n\n')
        docs_not_split = chunks.get("docs_not_split")
        for c in docs_not_split:
            f.write(
                f'\n{"-" * 50} CHUNK ID: {c.metadata.get("chunk_id")} {"-" * 50}\n'
            )
            f.write(f"{c.page_content}\n\n")


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )

