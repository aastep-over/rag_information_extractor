from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Python native
import asyncio, os
from pathlib import Path
from typing import List, Tuple, Literal
import json

# from other modules
from rag_info_extractor.document_ingestion.load_docs import aload_pdfs
from rag_info_extractor.utils.embedder import HFEmbedder

# logging relative
import logging
logger = logging.getLogger(__name__)


def load_docs_from_dir(
    dataset_dir: str | Path,
    chunks_type: Literal["fixed_size_chunks", "custom_chunks", "semantic_chunks"],
    HF_embedding_model_name: str,
    evaluator_llm: str,
    llm_model: str,
    max_embed_tokens: int,
    read_mode: str,
    pages_joining_str: str
) -> Tuple[List[Document], List[Document]]:
    
    # Load doc and split in chunks
    loaded_docs = asyncio.run(aload_pdfs(
        folder = dataset_dir,
        HF_embedding_model_name = HF_embedding_model_name,
        evaluator_llm=evaluator_llm,
        llm_model=llm_model,
        max_embed_tokens=max_embed_tokens,
        num_pdfs = 4,
        split = True,
        chunk_size = 430,
        chunk_overlap = 105,
        chunks_type = chunks_type,
        read_mode = read_mode,
        pages_joining_str = pages_joining_str
    ))

    parent_chunks, children_chunks = loaded_docs.get("parent_chunks", []), loaded_docs.get("children_chunks", [])

    return parent_chunks, children_chunks



def save_parent_chunk_bytes(
    parent_chunks: List[Document],
    doc_store_dir: str
):
    # Create folder
    os.makedirs(doc_store_dir, exist_ok=True)
    doc_store_page_content = LocalFileStore(f"{doc_store_dir}/page_content") # For persistent parent chunks
    doc_store_metadata = LocalFileStore(f"{doc_store_dir}/metadata")

    # Save parent chunks in bytes files
    parents_kv: List[tuple[int, Document]] = [(p.metadata.get("chunk_id", ""), p) for p in parent_chunks]
    doc_store_page_content.mset([(f"{str(k)}", bytes(d.page_content, encoding="utf-8")) for k, d in parents_kv])  # page_content
    doc_store_metadata.mset([(f"{str(k)}", bytes(json.dumps(d.metadata, ensure_ascii=False, indent=4), encoding="utf-8")) for k, d in parents_kv])

    logger.info("Parent Chunks stored.")

def embed_child_chunks(
    children_chunks: List[Document],
    HF_embedding_model_name: str,
    vector_store_dir: str
):
    # Create folder
    os.makedirs(vector_store_dir, exist_ok=True)

    # Save children chunks in vector db
    
    embedding_func = HFEmbedder(normalize_embeddings=True) # TODO: Enforce model checking when initialized in class HFEmbedder
    try:
        # embedding_func = HuggingFaceEmbeddings(
        #     model_name = HF_embedding_model_name, # HuggingFace embedding model
        #     encode_kwargs = {"normalize_embeddings": True}
        # )
        vector_store = Chroma(
            embedding_function = embedding_func,
            persist_directory = vector_store_dir,
            collection_name = "pdf_chunks"
        )
        asyncio.run(vector_store.aadd_documents(children_chunks))
        
        logger.info(f"Embeddings Created and stored in {vector_store_dir}")
    except OSError:
        logger.exception("ERROR! (Unable to load embedding model). Enter model name from HuggingFace")






if __name__ == "__main__":

    # Load args form config file
    import os
    import yaml
    import time
    import argparse
    from dotenv import load_dotenv

    # logging relative
    import logging
    from rag_info_extractor.utils.common_logging import configure_logging
    from rag_info_extractor.utils.load_config import cfgs 
    
    logger = logging.getLogger(__name__)

    t0 = time.time()

    # # CONFIG FILE SETTINGS
    cfgs = cfgs.get("args", {})

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
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, f"{CHUNKS_TYPE}")  # TODO: REMOVE "__0.8", its a setting (BREAKPOINT_THRESHOLD) in src/rag_info_extractor/document_ingestion/load_docs.py 
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, f"{CHUNKS_TYPE}") # TODO: REMOVE "__0.8" 

    # Load env_vars
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    EMBEDDING_MODEL_NAME_ENV = EMBEDDING_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
    EMBEDDING_MODEL_PATH = os.environ.get(EMBEDDING_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME)

    # configure logging
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO, logfile=os.path.join(BASE_DIR, "info_extractor.log"))        # default INFO or env override
    logger.info(f'Loading the documents: {", ".join(os.listdir(DATASET_DIR))}')

    # load and split docs in parent & child chunks
    parent_chunks, children_chunks = load_docs_from_dir(
        dataset_dir = DATASET_DIR,
        chunks_type = CHUNKS_TYPE,
        HF_embedding_model_name = EMBEDDING_MODEL_PATH,
        evaluator_llm = EVALUATOR_LLM,
        llm_model = LLM_MODEL,
        max_embed_tokens = MAX_EMBED_TOKENS,
        read_mode = READ_MODE,
        pages_joining_str = PAGES_JOINING_STR
    )

    logger.info(f"Loaded {len(parent_chunks)} Parent chunks from {len(set(d.metadata['filename'] for d in parent_chunks))} PDFs")
    logger.info(f"Loaded {len(children_chunks)} Children chunks from {len(set(d.metadata['filename'] for d in children_chunks))} PDFs")


    # save parent chunks and embed child chunks
    if parent_chunks:
        save_parent_chunk_bytes(
            parent_chunks,
            doc_store_dir = DOC_STORE_LARGE_CHUNKS_PATH
        )
    if children_chunks:
        embed_child_chunks(
            children_chunks,
            HF_embedding_model_name = EMBEDDING_MODEL_PATH,
            vector_store_dir = VECTOR_STORE_PATH
        )


    logger.info(f'Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}')

    

