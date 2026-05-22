import asyncio

# logging relative
import logging
import os

# Python native
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def fixed_size_chunking(
    child_splitter,
    docs: List[Document],
    tokenizer,
    last_parent_id: int = -1,
    last_child_id: int = -1,
) -> Dict[str, List[Document]]:

    # Build parent splitter (sync, cheap)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4 * getattr(child_splitter, "_chunk_size", 1000),
        chunk_overlap=4 * getattr(child_splitter, "_chunk_overlap", 0),
        add_start_index=True,
    )

    # Create Parent chunks
    parent_chunks: List[Document] = []
    for d in docs:
        parent_chunks.extend(parent_splitter.split_documents([d]))

    # Add chunk ids to parent_chunks
    for i, chunk in enumerate(parent_chunks):
        chunk.metadata["chunk_id"] = (last_parent_id + 1) + i

    # Create Child chunks from parent chunks
    children_chunks: List[Document] = []
    child_id = last_child_id + 1
    for p in parent_chunks:
        sub = child_splitter.split_documents([p])
        for ch in sub:
            # calculate extra info
            start = ch.metadata.get("start_index", None)
            n_chars = len(ch.page_content or "")
            n_toks = len(tokenizer.encode(ch.page_content or ""))
            ch.metadata.update(
                {
                    "parent_id": p.metadata.get(
                        "chunk_id"
                    ),  # Necessary for parent chunk
                    "chunk_id": child_id,
                    "n_chars": n_chars,
                    "n_tokens": n_toks,
                }
            )
            child_id += 1
            if start is not None:
                ch.metadata["char_start"] = int(start)
                ch.metadata["char_end"] = int(start) + n_chars
            children_chunks.append(ch)

    return {
        "parent_chunks": parent_chunks,
        "children_chunks": children_chunks,
    }


async def fixed_size_chunking_async(
    child_splitter,
    docs: List[Document],
    tokenizer,
    max_concurrency: int = 8,
    last_parent_id: int = -1,
    last_child_id: int = -1,
) -> Dict[str, List[Document]]:
    """
    Async fixed-size chunking:
    - Builds parent chunks with a larger splitter.
    - Builds child chunks from each parent.
    - Computes metadata (doc_id, chunk_index, n_chars, n_tokens, char_start/char_end).
    All blocking operations are executed off-thread.
    """

    # Build parent splitter (sync, cheap)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5 * getattr(child_splitter, "_chunk_size", 2000),
        chunk_overlap=5 * getattr(child_splitter, "_chunk_overlap", 400),
        add_start_index=True,
    )

    # --- Split parents concurrently ---
    sem = asyncio.Semaphore(max_concurrency)

    async def _split_parent(d: Document) -> List[Document]:
        """Run parent splitting off-thread to avoid blocking the event loop."""
        async with sem:
            # split_documents expects a List[Document]
            return await asyncio.to_thread(parent_splitter.split_documents, [d])

    parent_lists = await asyncio.gather(*[_split_parent(d) for d in docs])
    parent_chunks: List[Document] = [pc for sub in parent_lists for pc in sub]

    # don't include chunks with only text used for defining changing pages
    page_splitter_text = (
        "-" * 30 + "THIS IS A CUSTOM END OF PAGE" + "-" * 30
    )  # joinging_str
    filtered_parents: List[Document] = [
        d
        for d in parent_chunks
        if (d.page_content not in page_splitter_text) and (d.page_content != "")
    ]
    # Add chunk id to parent chunks
    for i, chunk in enumerate(filtered_parents):
        chunk.metadata["chunk_id"] = (last_parent_id + 1) + i
        chunk.metadata["pattern_name"] = "fixed_size"

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
            subs: List[Document] = await asyncio.to_thread(
                child_splitter.split_documents, [p]
            )

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

    children_lists = await asyncio.gather(
        *[_split_children_from_parent(p) for p in filtered_parents]
    )
    children_chunks: List[Document] = [c for sub in children_lists for c in sub]

    return {
        "parent_chunks": filtered_parents,
        "children_chunks": children_chunks,
    }
