from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer

# Python native
from pathlib import Path
import asyncio
from typing import List, Dict, Optional
import os

# logging relative
import logging
logger = logging.getLogger(__name__)


def semantic_chunking(
    docs: List[Document],
    embedding_func,
    tokenizer, 
    splitter,
    max_embed_tokens: int,
    pages_joining_str: Optional[str],
    last_parent_id: int = -1,
    last_child_id: int = -1,
    breakpoint_threshold_amount: Optional[float] = None

) -> Dict[str, List[Document]]:

    # Initialize and create semantic chunks
    semantic_chunker = SemanticChunker(embedding_func, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=breakpoint_threshold_amount) # breakpoint_threshold_amount
    parent_chunks = semantic_chunker.create_documents(
        texts = [d.page_content for d in docs],
        metadatas = [d.metadata for d in docs]
    )

    # Remove chunks which contain only pages_joining_str or are empty
    parent_chunks = [pc for pc in parent_chunks if (pc.page_content not in pages_joining_str if pages_joining_str else True) and (pc.page_content != "")]

    # Add chunk ids to parent_chunks
    for i, chunk in enumerate(parent_chunks):
        chunk.metadata["chunk_id"] = (last_parent_id + 1) + i

    # Create child chunks by splitting the parent if > token_limit
    children_chunks: List[Document] = []
    child_id = (last_child_id + 1)
    for p in parent_chunks:
        p_token_size = len(tokenizer.encode(p.page_content or ""))

        if p_token_size <= max_embed_tokens:
            ch = Document(page_content=p.page_content, metadata=p.metadata)
            start = ch.metadata.get("start_index", None)
            n_chars = len(ch.page_content or "")
            ch.metadata.update({
                "parent_id": p.metadata.get("chunk_id"),          # Necessary for parent chunk
                "chunk_id": child_id,
                "n_chars": n_chars,
                "n_tokens": p_token_size,
                "pattern_name": "semantic"
            })
            if start is not None:
                    ch.metadata["char_start"] = int(start)
                    ch.metadata["char_end"] = int(start) + n_chars

            child_id += 1
            children_chunks.append(ch)
        else:
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
    embedding_func,
    tokenizer,
    text_splitter,
    max_embed_tokens: int,
    pages_joining_str: Optional[str], 
    max_concurrency: int = 8,
    last_parent_id: int = -1,
    last_child_id: int = -1,
    breakpoint_threshold_amount: Optional[float] = None

) -> Dict[str, List[Document]]: 

    # Initialize and create semantic chunks
    semantic_chunker = SemanticChunker(embedding_func, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=breakpoint_threshold_amount)

    # --- Split parents concurrently ---
    sem = asyncio.Semaphore(max_concurrency)

    async def _chunk_one(d: Document) -> List[Document]:
        """Run semantic parent splitting off-thread to avoid blocking the event loop."""
        async with sem:
            # split_documents expects a List[Document]
            return await asyncio.to_thread(semantic_chunker.create_documents, [d.page_content], [d.metadata])
    
    per_doc_chunks = await asyncio.gather(*[_chunk_one(d) for d in docs])
    parent_chunks: List[Document] = [pc for sub in per_doc_chunks for pc in sub if (pc.page_content not in pages_joining_str if pages_joining_str else True) and (pc.page_content != "") ]

    # Add chunk id to parent chunks
    for i, chunk in enumerate(parent_chunks):
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
            # 1) Split into child chunks if p_tokens > max_embed_tokens(blocking)
            p_tokens = len(tokenizer.encode(p.page_content or ""))

            if p_tokens > max_embed_tokens:
                subs: List[Document] = await asyncio.to_thread(text_splitter.split_documents, [p])
            else:
                subs = [Document(page_content=p.page_content, metadata=p.metadata)]

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

    children_lists = await asyncio.gather(*[_split_children_from_parent(p) for p in parent_chunks])
    children_chunks: List[Document] = [c for sub in children_lists for c in sub]

    return {
        "parent_chunks": parent_chunks,
        "children_chunks": children_chunks,
    }
