# Python native
import asyncio
import json

# logging relative
import logging
import os
from pathlib import Path
from typing import List, Literal, Tuple

from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# from other modules
from rag_info_extractor.document_ingestion.load_docs import aload_pdfs
from rag_info_extractor.utils.embedder import HFEmbedder

logger = logging.getLogger(__name__)


def load_docs_from_dir(
    dataset_dir: str | Path,
    chunks_type: Literal["fixed_size_chunks", "custom_chunks", "semantic_chunks"],
    HF_embedding_model_name: str,
    evaluator_llm: str,
    llm_model: str,
    max_embed_tokens: int,
    read_mode: str,
    pages_joining_str: str,
) -> Tuple[List[Document], List[Document]]:

    # Load doc and split in chunks
    loaded_docs = asyncio.run(
        aload_pdfs(
            folder=dataset_dir,
            HF_embedding_model_name=HF_embedding_model_name,
            evaluator_llm=evaluator_llm,
            llm_model=llm_model,
            max_embed_tokens=max_embed_tokens,
            num_pdfs=4,
            split=True,
            chunk_size=430,
            chunk_overlap=105,
            chunks_type=chunks_type,
            read_mode=read_mode,
            pages_joining_str=pages_joining_str,
        )
    )

    parent_chunks, children_chunks = loaded_docs.get(
        "parent_chunks", []
    ), loaded_docs.get("children_chunks", [])

    return parent_chunks, children_chunks


def save_parent_chunk_bytes(parent_chunks: List[Document], doc_store_dir: str):
    # Create folder
    os.makedirs(doc_store_dir, exist_ok=True)
    doc_store_page_content = LocalFileStore(
        f"{doc_store_dir}/page_content"
    )  # For persistent parent chunks
    doc_store_metadata = LocalFileStore(f"{doc_store_dir}/metadata")

    # Save parent chunks in bytes files
    parents_kv: List[tuple[int, Document]] = [
        (p.metadata.get("chunk_id", ""), p) for p in parent_chunks
    ]
    doc_store_page_content.mset(
        [(f"{str(k)}", bytes(d.page_content, encoding="utf-8")) for k, d in parents_kv]
    )
    doc_store_metadata.mset(
        [
            (
                f"{str(k)}",
                bytes(
                    json.dumps(d.metadata, ensure_ascii=False, indent=4),
                    encoding="utf-8",
                ),
            )
            for k, d in parents_kv
        ]
    )

    logger.info("Parent Chunks stored.")


def embed_child_chunks(children_chunks: List[Document], vector_store_dir: str):
    # Create folder
    os.makedirs(vector_store_dir, exist_ok=True)

    # Save children chunks in vector db

    embedding_func = HFEmbedder(
        normalize_embeddings=True
    )  # TODO: Enforce model checking when initialized in class HFEmbedder
    try:
        vector_store = Chroma(
            embedding_function=embedding_func,
            persist_directory=vector_store_dir,
            collection_name="pdf_chunks",
        )
        asyncio.run(vector_store.aadd_documents(children_chunks))

        logger.info(f"Embeddings Created and stored in {vector_store_dir}")
    except OSError:
        logger.exception(
            "ERROR! (Unable to load embedding model). Enter model name from HuggingFace"
        )
