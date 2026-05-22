import os, time, datetime
from pathlib import Path
from rag_info_extractor.utils.embedder import HFEmbedder
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from rag_info_extractor.utils.common_logging import configure_logging
import argparse
from rag_info_extractor.utils.load_config import cfgs
import logging
import asyncio
import json
from dotenv import load_dotenv

from rag_info_extractor.rag_pipeline_components.retrieve import retrieve
from rag_info_extractor.rag_pipeline_components.re_ranker import cross_encode_rerank, faster_retrieve_and_rerank, across_encode_rerank, afaster_retrieve_and_rerank
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def main(functions_to_run: list[str]):
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
    logger.info(f"Logging for {'-'*30} rag_information_extractor/src/rag_info_extractor/rag_pipeline/retrieve.py")

    # 2. CONFIG FILE SETTINGS:
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = Path(__file__).resolve().parents[3]
    
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) 
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE) 

    # 4. Setup Query and Aziende (EXAMPLE)
    QUESTION = "Agli amministratori spetta il rimborso delle spese?"
    QUERY = "Agli amministratori spetta il rimborso delle spese?"
    AZIENDA = '2kind srl' # azienda_name_records[0]
    
    logger.info(f"Docs loaded from {DOC_STORE_LARGE_CHUNKS_PATH}")
    # Load Vector and Doc store
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
    AZIENDA = azienda_name_records[0]

    # Run Retrieval to get contexts
    logger.info("Retrieving docs...")
    retrieval_output = retrieve(
        retriever = retriever,
        query = QUERY,
        doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
        azienda = AZIENDA,
        pages_joining_str = PAGES_JOINING_STR,
        retrieve_parents = False
    )
    logger.info(f"Retrived docs (IDs): {retrieval_output.get('retrieved_docs_ids')}")
    
    # Run CrossEncode Re-ranker
    if "cross_encode_rerank" in functions_to_run:
        logger.info("Re-Ranking docs via Cross-Encode-Reranker...")
        output = cross_encode_rerank(
            contexts = retrieval_output.get("context", []),
            question = QUESTION,
            doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
            k_min = 2,
            k_max = 5,
            rel_thresh = 0.4,
            max_promoted_parents = 3,
            use_parent_heuristics = False,
            save_full_chunks = False
        )
    # Async version of CrossEncoder Re-ranker
    if "across_encode_rerank" in functions_to_run:
        logger.info("Re-Ranking docs via (async) Cross-Encode-Reranker...")
        output = asyncio.run(
            across_encode_rerank(
                contexts = retrieval_output.get("context", []),
                question = QUESTION,
                doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
                k_min = 2,
                k_max = 5,
                rel_thresh = 0.4,
                max_promoted_parents = 3,
                use_parent_heuristics = False
            )
        )

    # Run faster (cross-encode) reranker
    if "faster_retrieve_and_rerank" in functions_to_run:
        # fast_re_ranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL), top_n=8) # re_ranker compressor for fast retrieve + re_rank+ compression
        logger.info("Re-Ranking docs via Faster-Retrieve+Rerank...")
        fast_reranker_output = faster_retrieve_and_rerank(
            query = QUERY,
            retriever = retriever,
            azienda = AZIENDA,
            pages_joining_str = PAGES_JOINING_STR,
            top_n = 4
        )
    # Async version of faster re-ranker
    if "afaster_retrieve_and_rerank" in functions_to_run:
        logger.info("Re-Ranking docs via (async) Faster-Retrieve+Rerank...")
        fast_reranker_output = asyncio.run(
            afaster_retrieve_and_rerank(
                query = QUERY,
                retriever = retriever,
                azienda = AZIENDA,
                pages_joining_str = PAGES_JOINING_STR,
                top_n = 4
            )
        )

    # Write output to a file
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write(f"## OUTPUT FOR: re_ranker.py \n{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Save outputs of cross_encode_reranker
        if ("cross_encode_rerank" in functions_to_run) or ("across_encode_rerank" in functions_to_run):
            f.write(f"# FUNCTION: {'cross_encode_rerank' if 'cross_encode_rerank' in functions_to_run else '(async) cross_encode_rerank'}\n")
            f.write(f"QUESTION: {QUESTION}\n")
            f.write("Re-Ranked Documents: \n")
            re_ranked_docs = output.get("context", [])
            for i, c in enumerate(re_ranked_docs):
                f.write(f'\n{"-"*50} CHUNK {i} {"-"*50}\n')
                f.write(f'CHUNK ID: {c.metadata.get("chunk_id")}\n')
                f.write(f"{c.page_content}\n\n")

            f.write(f'{"x"*100}\n')

            f.write("Re-Ranked Doc IDs: \n\n")
            f.write(f'parents =  {output.get("re_ranked_docs_ids").get("parents")}\n')
            f.write(f'children =  {output.get("re_ranked_docs_ids").get("children")}\n\n\n')
            
            f.write(f'{"x"*100}\n')

            f.write("Re-Ranked Doc Texts: \n\n")
            f.write(f'parents =  {output.get("re_ranked_docs_texts").get("parents")}\n')
            f.write(f'children =  {output.get("re_ranked_docs_texts").get("children")}\n')

            f.write(f'{"x"*100}\n')

            f.write("Debug Info: \n\n")
            json.dump(output.get("re_rank_debug", {}), f, indent=4)


        # Save outputs of faster_retrieve_and_rerank
        if ("faster_retrieve_and_rerank" in functions_to_run) or ("afaster_retrieve_and_rerank" in functions_to_run):
            f.write(f'\n{"x"*100}\n')
            f.write(f'{"x"*100}\n\n')

            f.write(f"# FUNCTION: {'faster_retrieve_and_rerank' if 'faster_retrieve_and_rerank' in functions_to_run else '(async) faster_retrieve_and_rerank'}\n")
            f.write(f"QUERY: {QUERY}\n")
            f.write("Documents (Retrieved + Re-ranked + compressed): \n")
            docs_retrieved_re_ranked = fast_reranker_output.get("context")
            for i, c in enumerate(docs_retrieved_re_ranked):
                f.write(f'\n{"-"*50} CHUNK {i} {"-"*50}\n')
                f.write(f'CHUNK ID: {c.metadata.get("chunk_id")}\n')
                f.write(f"{c.page_content}\n\n")
            
            f.write(f'{"x"*100}\n')
            f.write(f'Doc IDs: {fast_reranker_output.get("docs_ids")}\n\n')

            f.write(f'{"x"*100}\n')
            f.write(f'Re-Ranked Doc Texts: {fast_reranker_output.get("docs_texts")}\n\n')    


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})

    # Run and save outputs of each function
    functions_to_run = ["across_encode_rerank"] # in ["cross_encode_rerank", "faster_retrieve_and_rerank"]  # can be both (and their async versions)
    main(functions_to_run)

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )