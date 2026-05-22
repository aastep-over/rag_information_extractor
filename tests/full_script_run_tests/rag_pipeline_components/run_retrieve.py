import os, time, datetime
from pathlib import Path
from langchain_chroma import Chroma
from rag_info_extractor.utils.common_logging import configure_logging
import argparse
from rag_info_extractor.utils.load_config import cfgs
from rag_info_extractor.utils.embedder import HFEmbedder

import logging
import asyncio

from rag_info_extractor.rag_pipeline_components.retrieve import retrieve, aretrieve

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
    logger.info(f"Logging for {"-"*30} rag_information_extractor/src/rag_info_extractor/rag_pipeline/retrieve.py")

    # 2. CONFIG FILE SETTINGS
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = Path(__file__).resolve().parents[3]
    
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) 
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE)
    

    # 3. Load Vector and Doc store
    embedding = HFEmbedder(normalize_embeddings=True)
    vector_store = Chroma(embedding_function=embedding,
                        persist_directory=VECTOR_STORE_PATH,
                        collection_name="pdf_chunks")
    retriever = vector_store.as_retriever(search_type="similarity",
                                        search_kwargs={'k': 8})

    # Get all azienda names in vector/doc store
    nome_delle_aziende = set((vector_store.get()['metadatas'][i].get('azienda'), vector_store.get()['metadatas'][i].get('filename')) for i in range(len(vector_store.get()['ids']))) 
    nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    azienda_name_records = [x[0] for x in nome_delle_aziende]

    # 4. Set up Query and Aziende (EXAMPLE)
    QUERY = "Agli amministratori spetta il rimborso delle spese?"
    AZIENDA = azienda_name_records[0]

    # 5. Run Retrieval
    logger.info("Retrieving docs...")
    if RUN_ASYNC:
        logger.info("Async version...")
        output = asyncio.run(
            aretrieve(
                retriever = retriever,
                query = QUERY,
                doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
                azienda = AZIENDA,
                pages_joining_str = PAGES_JOINING_STR,
                retrieve_parents = True
            )
        )
    else:
        logger.info("Sync version...")
        output = retrieve(
            retriever = retriever,
            query = QUERY,
            doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
            azienda = AZIENDA,
            pages_joining_str = PAGES_JOINING_STR,
            retrieve_parents = True
        )


    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("## OUTPUT FOR: retrieve.py\n")
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write(f"# Async: {'YES' if RUN_ASYNC else 'NO'}\n\n")
        f.write(f"QUERY: {QUERY}\n")
        f.write(f"AZIENDA: {AZIENDA}\n")
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


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})
    RUN_ASYNC = True

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )