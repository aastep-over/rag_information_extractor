import argparse
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List

import fitz
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_info_extractor.document_ingestion.fixed_size_chunking import (
    fixed_size_chunking,
    fixed_size_chunking_async,
)
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.load_config import cfgs
from transformers import AutoTokenizer

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
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)

    # 2. CONFIG FILE SETTINGS:
    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
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

    # 4. Load pdf
    docs: List[Document] = []
    if PDF_LOADER == "pymupdf":
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info(f"Document exists: {os.path.exists(path)}")

            with open(str(path), "rb") as fh:
                data = fh.read()
            doc = fitz.open(stream=data, filetype="pdf")

            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            meta = {
                **meta,
                **{
                    "source": Path(doc.name).name if doc.name else None,
                    "total_pages": doc.page_count,
                },
            }
            joining_str = "-" * 30 + "THIS IS A CUSTOM END OF PAGE" + "-" * 30
            text = joining_str.join(page.get_text() if isinstance(page.get_text("text"), str) else "" for page in doc)  # type: ignore (return type of get_text is not only str)
            docs.append(Document(page_content=text, metadata=meta))

    else:
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info(f"Document exists: {os.path.exists(path)}")
            loader = PyPDFLoader(
                path,
                mode="single",
                pages_delimiter="-" * 30 + "THIS IS A CUSTOM END OF PAGE" + "-" * 30,
            )
            docs.extend(loader.load())

    logger.info("Docs Loaded")

    # 5. Define text splitter and tokenizer
    chunk_size = 400
    chunk_overlap = 100
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, use_fast=True)
    child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,  # chunk size (tokens)
        chunk_overlap=chunk_overlap,  # chunk overlap (tokens)
        add_start_index=True,  # track index in original document
    )

    # 6. Run the chunking function
    logger.info("Creating Chunks...")
    if RUN_ASYNC:
        logger.info("Async...")
        chunks = asyncio.run(
            fixed_size_chunking_async(
                child_splitter,
                docs,
                tokenizer,
                max_concurrency=8,
                last_parent_id=-1,
                last_child_id=-1,
            )
        )
    else:
        logger.info("Sync...")
        chunks = fixed_size_chunking(
            child_splitter, docs, tokenizer, last_parent_id=-1, last_child_id=-1
        )

    # 7. Obtain and sort parent and children chunks and store them in a temp txt file
    parent_chunks, children_chunks = chunks.get("parent_chunks", []), chunks.get(
        "children_chunks", []
    )
    parent_chunks = sorted(parent_chunks, key=lambda x: x.metadata.get("chunk_id", -1))
    children_chunks = sorted(
        children_chunks, key=lambda x: x.metadata.get("chunk_id", -1)
    )

    logger.info("Chunks created. Saving...")
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("OUTPUT FOR fixed_size_chunking.py\n\n")
        f.write("Fixed-Sized Chunks: \n")

        f.write("PARENT CHUNKS: \n\n")
        for c in parent_chunks:
            f.write(f"\n{"-"*50} CHUNK ID: {c.metadata.get("chunk_id")} {"-"*50}\n")
            f.write(f"{c.page_content}\n\n")

        # f.writelines([f"CHUNK ID = {c.metadata.get("chunk_id")}\n{c.page_content}\n\n" for c in parent_chunks])
        f.write(f"{"x"*100}\n")

        f.write("Children Chunks\n")
        for c in children_chunks:
            f.write(f"\n{"-"*50} CHUNK ID: {c.metadata.get("chunk_id")} {"-"*50}\n")
            f.write(f"{c.page_content}\n\n")
        # f.writelines([f"CHUNK ID = {c.metadata.get("chunk_id")}\n{c.page_content}\n\n" for c in children_chunks])


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})
    RUN_ASYNC = True

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )
