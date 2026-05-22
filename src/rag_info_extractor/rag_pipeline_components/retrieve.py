import asyncio
import json

# Logging
import logging

# Python native
from typing import Dict, List, Optional, TypedDict

import aiofiles
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever

logger = logging.getLogger(__name__)


class _Output_retrieve(TypedDict):
    context: List[Document]
    retrieved_docs_ids: Dict[str, List[int]]
    retrieved_docs_texts: Dict[str, List[str]]


def retrieve(
    retriever: VectorStoreRetriever,
    query: str,
    doc_store_large_chunks_path: Optional[str],
    azienda: str = "",
    pages_joining_str: Optional[str] = None,
    retrieve_parents: bool = False,
    save_full_chunks: bool = False,
) -> _Output_retrieve:

    logger.info("\n --------------- NODE: __retrieve__ ------------------------\n")  ###

    # If no azienda name specified, fetch for all
    if not azienda:
        retrieved_docs = retriever.invoke(
            query,
        )
    else:
        retrieved_docs = retriever.invoke(
            query,
            filter={"azienda": azienda},
        )

    # Retrieve Parent chunks if required
    if retrieve_parents and doc_store_large_chunks_path:
        parent_keys = set([d.metadata["parent_id"] for d in retrieved_docs])
        logger.debug("Keys of parent chunks inside retriever node:", parent_keys)
        page_contents = ["" for i in range(len(parent_keys))]
        metadatas = [{} for i in range(len(parent_keys))]

        for i, id in enumerate(parent_keys):
            with open(
                f"{doc_store_large_chunks_path}/page_content/{id}", encoding="utf-8"
            ) as f:
                page_contents[i] = f.read()
            with open(
                f"{doc_store_large_chunks_path}/metadata/{id}", encoding="utf-8"
            ) as f:
                metadatas[i] = json.load(f)

        if page_contents and metadatas:
            retrieved_docs: List[Document] = [
                Document(page_content=p, metadata=m)
                for p, m in zip(page_contents, metadatas)
                if (p and m)
            ]

    # remove page joining str to avoid confusion for llm in generation phase
    if pages_joining_str:
        for d in retrieved_docs:
            d.page_content = d.page_content.replace(pages_joining_str, "\n")

    # ------------- for bookkeeping & return -----------------------------------
    parents_retrieved_docs_ids = []
    children_retrieved_docs_ids = []
    parents_retrieved_docs_texts = []
    children_retrieved_docs_texts = []
    for d in retrieved_docs:
        if "parent_id" in d.metadata.keys():
            children_retrieved_docs_ids.append(d.metadata.get("chunk_id"))

            if save_full_chunks:
                children_retrieved_docs_texts.append(f"{d.page_content}")
            else:
                children_retrieved_docs_texts.append(
                    f"{d.page_content[:10]} ... {d.page_content[-10:]}"
                )
        else:
            parents_retrieved_docs_ids.append(d.metadata.get("chunk_id"))

            if save_full_chunks:
                parents_retrieved_docs_texts.append(f"{d.page_content}")
            else:
                parents_retrieved_docs_texts.append(
                    f"{d.page_content[:10]} ... {d.page_content[-10:]}"
                )

    retrieved_docs_ids = {
        "parents": parents_retrieved_docs_ids,
        "children": children_retrieved_docs_ids,
    }  ### retrieved_docs_parent
    retrieved_docs_texts = {
        "parents": parents_retrieved_docs_texts,
        "children": children_retrieved_docs_texts,
    }

    return _Output_retrieve(
        context=retrieved_docs,
        retrieved_docs_ids=retrieved_docs_ids,
        retrieved_docs_texts=retrieved_docs_texts,
    )


# Async version
async def aretrieve(
    retriever: VectorStoreRetriever,
    query: str,
    doc_store_large_chunks_path: Optional[str],
    azienda: str = "",
    pages_joining_str: Optional[str] = None,
    retrieve_parents: bool = False,
    save_full_chunks: bool = False,
) -> _Output_retrieve:

    logger.info(
        "\n --------------- NODE: (async) __retrieve__ ------------------------\n"
    )  ###

    # If no azienda name specified, fetch for all
    if not azienda:
        retrieved_docs = await retriever.ainvoke(
            query,
        )
    else:
        retrieved_docs = await retriever.ainvoke(
            query,
            filter={"azienda": azienda},
        )

    # Retrieve Parent chunks if required
    if retrieve_parents and doc_store_large_chunks_path:
        parent_keys = set([d.metadata["parent_id"] for d in retrieved_docs])
        logger.debug("Keys of parent chunks inside retriever node:", parent_keys)
        page_contents = ["" for i in range(len(parent_keys))]
        metadatas = [{} for i in range(len(parent_keys))]

        async def _load_parent_chunk(idx, id):
            async with aiofiles.open(
                f"{doc_store_large_chunks_path}/page_content/{id}",
                encoding="utf-8",
                mode="r",
            ) as f:
                page_contents[idx] = await f.read()

            async with aiofiles.open(
                f"{doc_store_large_chunks_path}/metadata/{id}",
                encoding="utf-8",
                mode="r",
            ) as f:
                json_content = await f.read()
                metadatas[idx] = json.loads(json_content)

        # store them as Document obj
        await asyncio.gather(
            *[_load_parent_chunk(i, id) for i, id in enumerate(parent_keys)]
        )

        if page_contents and metadatas:
            retrieved_docs: List[Document] = [
                Document(page_content=p, metadata=m)
                for p, m in zip(page_contents, metadatas)
                if (p and m)
            ]

    # remove page joining str to avoid confusion for llm in generation phase
    if pages_joining_str:
        for d in retrieved_docs:
            d.page_content = d.page_content.replace(pages_joining_str, "\n")

    # ------------- for bookkeeping & return -----------------------------------
    parents_retrieved_docs_ids = []
    children_retrieved_docs_ids = []
    parents_retrieved_docs_texts = []
    children_retrieved_docs_texts = []
    for d in retrieved_docs:
        if "parent_id" in d.metadata.keys():
            children_retrieved_docs_ids.append(d.metadata.get("chunk_id"))

            if save_full_chunks:
                children_retrieved_docs_texts.append(f"{d.page_content}")
            else:
                children_retrieved_docs_texts.append(
                    f"{d.page_content[:10]} ... {d.page_content[-10:]}"
                )
        else:
            parents_retrieved_docs_ids.append(d.metadata.get("chunk_id"))

            if save_full_chunks:
                parents_retrieved_docs_texts.append(f"{d.page_content}")
            else:
                parents_retrieved_docs_texts.append(
                    f"{d.page_content[:10]} ... {d.page_content[-10:]}"
                )

    retrieved_docs_ids = {
        "parents": parents_retrieved_docs_ids,
        "children": children_retrieved_docs_ids,
    }
    retrieved_docs_texts = {
        "parents": parents_retrieved_docs_texts,
        "children": children_retrieved_docs_texts,
    }

    return _Output_retrieve(
        context=retrieved_docs,
        retrieved_docs_ids=retrieved_docs_ids,
        retrieved_docs_texts=retrieved_docs_texts,
    )
