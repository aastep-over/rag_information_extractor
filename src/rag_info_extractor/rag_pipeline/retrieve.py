from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.storage import LocalFileStore

# Python native
from typing import List, Optional, Dict, TypedDict
import json

# Logging
import logging
logger = logging.getLogger(__name__)



class _Output_retrieve(TypedDict):
    context: List[Document]
    retrieved_docs_ids: Dict[str, List[int]]
    retrieved_docs_texts: Dict[str, List[str]]

def retrieve(
    retriever: VectorStoreRetriever,
    query: str,
    doc_store_page_content: Optional[LocalFileStore],
    doc_store_metadata: Optional[LocalFileStore],
    azienda: str = "",
    pages_joining_str: Optional[str] = None,
    retrieve_parents: bool = False
) -> _Output_retrieve:
    
    logger.info("\n --------------- NODE: __retrieve__ ------------------------\n")###

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
    if retrieve_parents and (doc_store_page_content and doc_store_metadata):
        parent_keys = set([d.metadata["parent_id"] for d in retrieved_docs])
        logger.debug("Keys of parent chunks inside retriever node:", parent_keys)
        page_contents: List[bytes | None] = doc_store_page_content.mget(list(map(str, parent_keys))) 
        metadatas: List[bytes | None] = doc_store_metadata.mget(list(map(str, parent_keys)))
        if page_contents and metadatas:
            retrieved_docs: List[Document] = [
                Document(
                    page_content=bytes.decode(p, encoding="utf-8"), 
                    metadata=json.loads(bytes.decode(m, encoding="utf-8"))
                )
                for p, m in zip(page_contents, metadatas) if (p and m)
            ]

    # remove page joining str to avoid confusion for llm
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
            children_retrieved_docs_ids.append(d.metadata.get('chunk_id'))
            children_retrieved_docs_texts.append(f"{d.page_content[:10]} ... {d.page_content[-10:]}")
        else:
            parents_retrieved_docs_ids.append(d.metadata.get('chunk_id'))
            parents_retrieved_docs_texts.append(f"{d.page_content[:10]} ... {d.page_content[-10:]}")

    retrieved_docs_ids = {
        "parents": parents_retrieved_docs_ids,
        "children": children_retrieved_docs_ids
    }### retrieved_docs_parent
    retrieved_docs_texts = {
        "parents": parents_retrieved_docs_texts,
        "children": children_retrieved_docs_texts
    }
    
    return _Output_retrieve(
        context = retrieved_docs,
        retrieved_docs_ids = retrieved_docs_ids,
        retrieved_docs_texts = retrieved_docs_texts
    )






if __name__ == "__main__":
    import yaml, os, time
    from pathlib import Path
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from rag_info_extractor.common_logging import configure_logging
    import argparse

    t0 = time.time()

    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Logging for {"-"*30} rag_information_extractor/src/rag_info_extractor/rag_pipeline/retrieve.py")

    # CONFIG FILE SETTINGS:
    cfg_path = Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    cfgs = configs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM") 
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) 
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE) 

    # Load Vector and Doc store
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                  encode_kwargs={"normalize_embeddings": True})
    vector_store = Chroma(embedding_function=embedding,
                        persist_directory=VECTOR_STORE_PATH,
                        collection_name="pdf_chunks")
    retriever = vector_store.as_retriever(search_type="similarity",
                                        search_kwargs={'k': 8})
    
    doc_store_page_content = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content") 
    doc_store_metadata = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/metadata")

    # Get all azienda names in vector/doc store
    nome_delle_aziende = set((vector_store.get()['metadatas'][i].get('azienda'), vector_store.get()['metadatas'][i].get('filename')) for i in range(len(vector_store.get()['ids']))) 
    nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    azienda_name_records = [x[0] for x in nome_delle_aziende]

    # Query and Aziende (EXAMPLE)
    QUERY = "Agli amministratori spetta il rimborso delle spese?"
    AZIENDA = azienda_name_records[0]

    # Run Retrieval
    logger.info("Retrieving docs...")
    output = retrieve(
        retriever = retriever,
        query = QUERY,
        doc_store_page_content = doc_store_page_content,
        doc_store_metadata = doc_store_metadata,
        azienda = AZIENDA,
        pages_joining_str = PAGES_JOINING_STR,
        retrieve_parents = False
    )


    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("## OUTPUT FOR: retrieve.py\n\n")
        f.write("DOCUMENTS RETRIEVED: \n")
        
        retrieved_docs = output.get("context", [])
        for i, c in enumerate(retrieved_docs):
            f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
            f.write(f"CHUNK ID: {c.metadata.get("chunk_id")}\n")
            f.write(f"{c.page_content}\n\n")

        f.write(f"{"x"*100}\n")

        f.write("RETRIEVED DOC IDs: \n\n")
        f.write(f"parents =  {output.get("retrieved_docs_ids").get("parents")}\n")
        f.write(f"children =  {output.get("retrieved_docs_ids").get("children")}\n\n\n")
        
        f.write(f"{"x"*100}\n")

        f.write("RETRIEVED DOC TEXTs: \n\n")
        f.write(f"parents =  {output.get("retrieved_docs_texts").get("parents")}\n")
        f.write(f"children =  {output.get("retrieved_docs_texts").get("children")}\n")
    
    print(output.get("context", []))

    logger.info(f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}")