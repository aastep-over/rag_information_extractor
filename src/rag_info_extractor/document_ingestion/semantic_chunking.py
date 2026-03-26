from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer

# Python native
from pathlib import Path
import asyncio
from typing import List, Dict
import os

# logging relative
import logging
logger = logging.getLogger(__name__)


def semantic_chunking(
    docs: List[Document],
    embedding_func: HuggingFaceEmbeddings,
    tokenizer, 
    splitter,
    last_parent_id: int = -1,
    last_child_id: int = -1

) -> Dict[str, List[Document]]:

    # Initialize and create semantic chunks
    semantic_chunker = SemanticChunker(embedding_func, breakpoint_threshold_type="percentile")
    parent_chunks = semantic_chunker.create_documents(
        texts = [d.page_content for d in docs],
        metadatas = [d.metadata for d in docs]
    )

    # Add chunk ids to parent_chunks
    for i, chunk in enumerate(parent_chunks):
        chunk.metadata["chunk_id"] = (last_parent_id + 1) + i

    # Create child chunks by splitting the parent if > token_limit
    children_chunks: List[Document] = []
    child_id = (last_child_id + 1)
    for p in parent_chunks:
        sub = splitter.split_documents([p])
        for ch in sub:
            # calculate extra info
            start = ch.metadata.get("start_index", None)
            n_chars = len(ch.page_content or "")
            n_toks = len(tokenizer.encode(ch.page_content or ""))
            ch.metadata.update({
                "parent_id": p.metadata.get("chunk_id"),          # Necessary for parent chunk
                "chunk_id": child_id,
                "n_chars": n_chars,
                "n_tokens": n_toks,
                "pattern_name": "semantic"
            })
            child_id += 1
            if start is not None:
                ch.metadata["char_start"] = int(start)
                ch.metadata["char_end"] = int(start) + n_chars
            children_chunks.append(ch)   

    return {
        "parent_chunks": parent_chunks,
        "children_chunks": children_chunks,
    }


async def semantic_chunking_async(
    docs: List[Document],
    embedding_func: HuggingFaceEmbeddings,
    tokenizer,
    splitter,
    max_concurrency: int = 8,
    last_parent_id: int = -1,
    last_child_id: int = -1

) -> Dict[str, List[Document]]: 

    # Initialize and create semantic chunks
    semantic_chunker = SemanticChunker(embedding_func, breakpoint_threshold_type="percentile")

    # --- Split parents concurrently ---
    sem = asyncio.Semaphore(max_concurrency)

    async def _split_parent(d: Document) -> List[Document]:
        """Run parent splitting off-thread to avoid blocking the event loop."""
        async with sem:
            # split_documents expects a List[Document]
            return await asyncio.to_thread(semantic_chunker.create_documents, [d.page_content], [d.metadata])
    
    parent_lists = await asyncio.gather(*[_split_parent(d) for d in docs])
    parent_chunks: List[Document] = [pc for sub in parent_lists for pc in sub]

    # don't include chunks with only text used for defining changing pages or is empty string
    page_splitter_text = ("-"*30 + "THIS IS A CUSTOM END OF PAGE" + "-"*30) # joinging_str
    filtered_parents: List[Document] = [d for d in parent_chunks if (d.page_content not in page_splitter_text) and (d.page_content != "")]
    # Add chunk id to parent chunks
    for i, chunk in enumerate(filtered_parents):
        chunk.metadata["chunk_id"] = (last_parent_id + 1) + i
        chunk.metadata["pattern_name"] = "semantic"


    # ----- Initialize child chunk id ------
    _child_id = last_child_id + 1 
    _child_id_lock = asyncio.Lock()

    async def _reserve_child_ids(n: int) -> int:
        """Atomically reserve n child ids and return the starting id."""
        async with _child_id_lock:
            nonlocal _child_id
            start = _child_id
            _child_id += n
            return start
    
    # --- Split children concurrently per parent ---
    async def _split_children_from_parent(p: Document) -> List[Document]:
        """Split one parent into children and compute metadata off-thread."""
        async with sem:
            # 1) Split into child chunks (blocking)
            subs: List[Document] = await asyncio.to_thread(splitter.split_documents, [p])

            # 2) Reserve a global id range for these children
            base = await _reserve_child_ids(len(subs))


            # 3) Compute per-child metadata (also off-thread to keep consistency)
            def _compute_child_meta() -> List[Document]:
                out: List[Document] = []
                for idx, ch in enumerate(subs):
                    text = ch.page_content or ""
                    start = ch.metadata.get("start_index", None)
                    n_chars = len(text)
                    n_toks = len(tokenizer.encode(text))
                    # Update metadata inplace
                    ch.metadata.update(
                        {
                            "parent_id": p.metadata.get("chunk_id"),
                            "chunk_id": base + idx,
                            "n_chars": n_chars,
                            "n_tokens": n_toks,
                        }
                    )
                    if start is not None:
                        ch.metadata["char_start"] = int(start)
                        ch.metadata["char_end"] = int(start) + n_chars
                    out.append(ch)
                return out

            return await asyncio.to_thread(_compute_child_meta)

    children_lists = await asyncio.gather(*[_split_children_from_parent(p) for p in filtered_parents])
    children_chunks: List[Document] = [c for sub in children_lists for c in sub]

    return {
        "parent_chunks": filtered_parents,
        "children_chunks": children_chunks,
    }

if __name__ == "__main__":

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    import fitz
    from pathlib import Path
    import yaml
    import time
    import argparse

    from rag_info_extractor.utils.common_logging import configure_logging
    from rag_info_extractor.utils.load_config import cfgs

    t0 = time.time()

    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)

    cfgs = cfgs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    PDF_LOADER = cfgs.get("PDF_LOADER", "./")
    
    DATASET_DIR = os.path.join(BASE_DIR, "data", "pdfs", DATASET_TYPE) 


    
    # Load pdf
    docs: List[Document] = []
    if PDF_LOADER == "pymupdf":
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info(f'Document exists: {os.path.exists(path)}')

            with open(str(path), "rb") as fh:
                data = fh.read()
            doc = fitz.open(stream=data, filetype="pdf")

            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            meta = {**meta, **{"source": Path(doc.name).name if doc.name else None, "total_pages": doc.page_count}}
            joining_str = "-"*30 + "THIS IS A CUSTOM END OF PAGE" + "-"*30
            text = joining_str.join(page.get_text() if isinstance(page.get_text("text"), str) else "" for page in doc ) # type: ignore (return type of get_text is not only str)
            docs.append(Document(page_content=text, metadata=meta))

    else:
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info(f'Document exists: {os.path.exists(path)}')
            loader = PyPDFLoader(path,
                                mode="single",
                                pages_delimiter="-"*30 + "THIS IS A CUSTOM END OF PAGE" + "-"*30
                                )
            docs.extend(loader.load())

    logger.info("Docs Loaded")
    # Define text splitter and tokenizer
    max_chunk_size = 430
    max_chunk_overlap = 105
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, use_fast=True)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=max_chunk_size,  # chunk size (tokens)
            chunk_overlap=max_chunk_overlap,  # chunk overlap (tokens)
            add_start_index=True,  # track index in original document
        )
    embedding_func = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Create Chunks
    logger.info("Creating Chunks...")
    # chunks = asyncio.run(fixed_size_chunking_async(
    #     child_splitter,
    #     docs,
    #     tokenizer,
    #     max_concurrency = 8,
    #     last_parent_id = -1,
    #     last_child_id = -1
    # ))
    chunks = semantic_chunking(
        docs,
        embedding_func,
        tokenizer,
        splitter,
        last_parent_id = -1,
        last_child_id = -1
    )


    # Obtain and sort parent and children chunks
    parent_chunks, children_chunks = chunks.get("parent_chunks", []), chunks.get("children_chunks", [])
    parent_chunks = sorted(parent_chunks, key=lambda x: x.metadata.get("chunk_id", -1))
    children_chunks = sorted(children_chunks, key=lambda x: x.metadata.get("chunk_id", -1))

    logger.info("Chunks created. Saving...")
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("OUTPUT FOR semantic_chunking.py\n\n")
        f.write("SEMANTIC-SIM-BASED Chunks: \n")
        
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
    
    logger.info(f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}")
    