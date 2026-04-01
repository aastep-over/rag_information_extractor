from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from sentence_transformers import CrossEncoder
from langchain_community.cross_encoders.base import BaseCrossEncoder
from pydantic import BaseModel, Field

# Python native
from typing import List, Optional, Dict, TypedDict, Tuple, Any
import json
import re
from collections import Counter

# from other modules
from rag_info_extractor.utils.apis_connector import call_reranker_service

# Logging
import logging
logger = logging.getLogger(__name__)





# ------------ HELPERS -----------------
# ---------------------------------------------------------------------------

# ------------ Span Density of child chunks and query-----------------
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

def _tokens(txt: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(txt or "")]

def _span_density(query: str, text: str) -> float:
    """Fraction of unique query terms (minus stop-ish tokens) present in text."""
    q = [t for t in _tokens(query) if len(t) > 2]
    if not q: 
        return 0.0
    qset = set(q)
    tset = set(_tokens(text))
    return len(qset & tset) / len(qset)

# ------------ Query Type (how/why/multi-hop)-----------------
def _query_type_flags(q: str) -> Dict[str, bool]:
    ql = q.lower()
    flags = {
        "is_how_why": any(w in ql for w in ["how ", "why ", "come ", "perché", "perche", "come si", "come posso"]),
        "is_policy_legal": any(w in ql for w in ["policy", "policies", "regulation", "regolamento", "statuto", "contratto", "obbligo", "obblighi", "compliance", "norma", "norme"]),
        "is_compare": any(w in ql for w in ["compare", "vs", "confronta", "differenza", "pro vs contro"]),
        "has_multi_hop_cues": any(w in ql for w in ["prima", "poi", "basandoti su", "according to", "based on"]),
        "is_long": len(q.split()) >= 18,
    }
    return flags

# ------------ Similarity between child chunks -----------------
def _similarity_margin(scores_sorted_desc: List[float], k_ref: int = 3) -> float:
    """Margin between best and kth (ratio). Smaller ratio → more ambiguity."""
    if not scores_sorted_desc:
        return 0.0
    top = scores_sorted_desc[0]
    kth = scores_sorted_desc[min(k_ref - 1, len(scores_sorted_desc) - 1)]
    if top <= 0:
        return 0.0
    return (top - kth) / max(top, 1e-9)  # normalized margin



# ---------------------------------------------------------------  MAIN FUNCTIONS  ---------------------------------------------------------------
# ---------------------------------------------------------------                  ---------------------------------------------------------------


## 1. ---------------------- CROSS-ENCODER RE-RANKER ----------------------------

class _Output_cross_encode_rerank(TypedDict):
    context: List[Document]
    re_ranked_docs_ids: Dict[str, List[int]]
    re_ranked_docs_texts: Dict[str, List[str]]
    re_rank_debug: Dict

def cross_encode_rerank(
    # re_ranker: CrossEncoder, 
    contexts: List[Document],
    question: str,
    doc_store_large_chunks_path: Optional[str],
    k_min: int = 2,
    k_max: int = 5,
    rel_thresh: float = 0.4,
    max_promoted_parents: int = 3,
    use_parent_heuristics: bool = False,
    save_full_chunks: bool = False

) -> _Output_cross_encode_rerank:
        """
        Re-rank the retrieved docs
        - k_min: min num of chunks to pass to generation
        - top_n: max num of chunks to pass to generation
        - rel_thres: keep docs with score >= rel_thres% of top score
        """
        logger.info("\n --------------- NODE: __cross_encode_rerank__ ------------------------\n")###

        # Reank only if context fetched by the retriever not empty (otherwise reranker.score(pairs) run in error due to empty list)
        if not contexts:
            return _Output_cross_encode_rerank(
                context = [],
                re_ranked_docs_ids = {},
                re_ranked_docs_texts = {},
                re_rank_debug = {}
            )
        

        query = question
        context_candidates = contexts

        # pairs = [(query, d.page_content) for d in context_candidates]
        # scores = re_ranker.predict(pairs, batch_size=2, show_progress_bar=False)
        scores = call_reranker_service(
            query = question,
            documents = [d.page_content for d in context_candidates]
        ) or []

        # sort indices by score (desc)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_scores = [scores[i] for i in order]
        top_score = sorted_scores[0]
        cut = top_score * rel_thresh if top_score > 0 else float("-inf")

        # choose min(max(k_min, count above threshold), k_max)
        keep_n = max(k_min, sum(1 for i in order if scores[i] >= cut))
        keep_n = min(keep_n, k_max)
        selected_idx = order[:keep_n]
        selected_docs = [context_candidates[i] for i in selected_idx] # children chunks 

        
        # ---------------- Heuristics block (to check if Parent chunk needed) ----------------
        if use_parent_heuristics:
            # 1) Parent concentration among the *top window* (use up to k_max or first 8)
            window_idx = order[:max(k_max, 8)]
            parents = []
            for i in window_idx:
                meta = context_candidates[i].metadata
                parents.append(meta.get("parent_id", f"__no_parent__:{i}"))  # robust default
            parent_counts = Counter(parents)
            if parent_counts:
                top_parent, top_parent_cnt = parent_counts.most_common(1)[0]
                parent_concentration = top_parent_cnt / max(len(window_idx), 1)
            else:
                top_parent, parent_concentration = None, 0.0
            
            # 2) Similarity margin among the first few
            sim_margin = _similarity_margin(sorted_scores, k_ref=min(3, len(sorted_scores)))

            # 3) Query type flags
            qflags = _query_type_flags(query)

            # 4) Span density (average across the selected children)
            densities = [_span_density(query, d.page_content) for d in selected_docs]
            avg_density = sum(densities) / max(len(densities), 1)

            # ---- Decision: should we promote parent(s)? ----
            # Tunable thresholds (start conservative, adjust with data):
            PARENT_CONC_T = 0.50      # ≥50% of the top window from same parent
            SIM_MARGIN_T  = 0.06      # <6% margin → scores are bunched (ambiguous)
            DENSITY_T     = 0.10      # <10% lexical coverage → likely need broader context

            need_parent = (
                (parent_concentration >= PARENT_CONC_T) or
                (sim_margin <= SIM_MARGIN_T) or
                (avg_density <= DENSITY_T) or
                qflags["is_how_why"] or
                qflags["is_policy_legal"] or
                qflags["is_compare"] or
                qflags["has_multi_hop_cues"] or
                qflags["is_long"]
            )

            promotion_notes = {
                "parent_concentration": round(parent_concentration, 3) or None,
                "similarity_margin": round(sim_margin, 3) or None,
                "avg_span_density": round(avg_density, 3) or None,
                "qflags": qflags,
                "need_parent": bool(need_parent)
            }
        else:
            need_parent = True
            promotion_notes = {
                "parent_concentration": None,
                "similarity_margin": None,
                "avg_span_density": None,
                "qflags": None,
                "need_parent": bool(need_parent)
            }

        if need_parent and doc_store_large_chunks_path:
            # Choose the most represented parents from the *selected set* to keep it tight
            sel_parent_counts = Counter([d.metadata.get("parent_id") for d in selected_docs])
            
            # do not include chunks which don't have a parent
            if None in sel_parent_counts.keys():
                sel_parent_counts.pop(None) 

            keys = [pid for pid, _ in sel_parent_counts.most_common(max_promoted_parents)] # List[int]

            page_contents = ["" for i in range(len(keys))]
            metadatas = [{} for i in range(len(keys))]

            for i, id in enumerate(keys):
                with open(f"{doc_store_large_chunks_path}/page_content/{id}", encoding="utf-8") as f:
                    page_contents[i] = f.read()
                with open(f"{doc_store_large_chunks_path}/metadata/{id}", encoding="utf-8") as f:
                    metadatas[i] = json.load(f)

            # store them as Document obj
            if page_contents and metadatas:
                promoted_parents_docs: List[Document] = [
                    Document(
                        page_content=p,
                        metadata=m
                    )
                    for p, m in zip(page_contents, metadatas) if (p and m)
                ]

            # De-duplicate by parent id; prefer the promoted parent over its children
            promoted_parent_ids = set([d.metadata.get("chunk_id") for d in promoted_parents_docs])
            filtered_children = [d for d in selected_docs 
                                if d.metadata.get("parent_id") not in promoted_parent_ids]
            # Compose final list: promoted parents first (preserve rank), then a few best children
            # Keep global cap ≈ k_max by trimming children
            remaining_slots = max(0, keep_n - len(promoted_parents_docs))
            re_ranked_docs = promoted_parents_docs[:max_promoted_parents] + filtered_children[:remaining_slots]

        else:
            re_ranked_docs = selected_docs

        # Optionally expose debug signals for tracing
        re_rank_debug = {}
        re_rank_debug["heuristics"] = promotion_notes
        re_rank_debug["selected_k"] = len(re_ranked_docs)

        

        # ------------- for bookkeeping & return -----------------------------------
        parents_re_ranked_docs_ids = []
        children_re_ranked_docs_ids = []
        parents_re_ranked_docs_texts = []
        children_re_ranked_docs_texts = []
        for d in re_ranked_docs:
            if "parent_id" in d.metadata.keys():
                children_re_ranked_docs_ids.append(d.metadata.get('chunk_id'))

                if save_full_chunks:
                    children_re_ranked_docs_texts.append(f"{d.page_content}")
                else:
                    children_re_ranked_docs_texts.append(f"{d.page_content[:10]} ... {d.page_content[-10:]}")
            else:
                parents_re_ranked_docs_ids.append(d.metadata.get('chunk_id'))

                if save_full_chunks:
                    parents_re_ranked_docs_texts.append(f"{d.page_content}")
                else:
                    parents_re_ranked_docs_texts.append(f"{d.page_content[:10]} ... {d.page_content[-10:]}")

        re_ranked_docs_ids = {
            "parents": parents_re_ranked_docs_ids,
            "children": children_re_ranked_docs_ids
        }
        re_ranked_docs_texts = {
            "parents": parents_re_ranked_docs_texts,
            "children": children_re_ranked_docs_texts
        }

        return _Output_cross_encode_rerank(
            context = re_ranked_docs,
            re_ranked_docs_ids = re_ranked_docs_ids,
            re_ranked_docs_texts = re_ranked_docs_texts,
            re_rank_debug = re_rank_debug
        )

# --------------------------      XXX                    ----------------------------


## 2. ---------------------- FAST RETRIEVER + RE-RANKER ----------------------------

class ReRankerBaseCE(BaseModel, BaseCrossEncoder):

    """Implement Re-ranker for fast_retrieve_and_rerank node"""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float] | None:
        """Compute similarity scores using a cross-encoder model.

        Args:
            text_pairs: The list of text text_pairs to score the similarity.

        Returns:
            List of scores, one for each pair.
        """
        sep_func = lambda x: (x[0][0], [x[i][1] for i in range(len(x))])
        query, documents = sep_func(text_pairs)
        scores = call_reranker_service(query, documents)
        
        return scores

class _Output_faster_retrieve_and_rerank(TypedDict):
    context: List[Document]
    docs_ids: Dict[str, List[int]]
    docs_texts: Dict[str, List[str]]

def faster_retrieve_and_rerank(
    query: str,
    # compressor: CrossEncoderReranker,
    retriever: VectorStoreRetriever,
    azienda: str = "",
    top_n: int = 4,
    pages_joining_str: Optional[str] = None,
    save_full_chunks: bool = False

) -> _Output_faster_retrieve_and_rerank:
        """
        Retrieve using similarity retriever, re-rank using reranker model and compress the re-ranked text.
        - top_n: max num of chunks to pass to generation
        """
        logger.info("\n --------------- NODE: faster__retrieve_and_rerank__ ------------------------\n")###

        compressor = CrossEncoderReranker(model=ReRankerBaseCE(), top_n=8)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
            )

        if azienda:
            docs = compression_retriever.invoke(
                query,
                filter = {"azienda": azienda},
                k = top_n
            )
        else:
            docs = compression_retriever.invoke(
                query,
                k = top_n
            )
        
        # Remove duplicate chunks
        ids_docs_already_included = []
        idx_to_include = []
        for i, d in enumerate(docs):
            if d.metadata.get("chunk_id") in ids_docs_already_included:
                continue
            else:
                idx_to_include.append(i)
                ids_docs_already_included.append(d.metadata.get("chunk_id"))
        
        docs = [docs[i] for i in idx_to_include]


        # remove page joining str to avoid confusion for llm
        if pages_joining_str:
            for d in docs:
                d.page_content = d.page_content.replace(pages_joining_str, "\n")

        
        
        # ------------- for bookkeeping & return -----------------------------------
        docs_ids = []
        docs_texts = []
        for d in docs:
            docs_ids.append(d.metadata.get('chunk_id'))

            if save_full_chunks:
                docs_texts.append(f"{d.page_content}")
            else:
                docs_texts.append(f"{d.page_content[:10]} ... {d.page_content[-10:]}")

        return _Output_faster_retrieve_and_rerank(
            context = docs,
            docs_ids = {
                "parents": [],
                "children": docs_ids
            },
            docs_texts = {
                "parents": [],
                "children": docs_texts
            }
        )





# ------------------------ For debugging the script ---------------------------------------------
def main(functions_to_run: List[str]):
    # Load Vector and Doc store
    # embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
    #                               encode_kwargs={"normalize_embeddings": True})
    embedding = HFEmbedder(normalize_embeddings=True)
    vector_store = Chroma(embedding_function=embedding,
                        persist_directory=VECTOR_STORE_PATH,
                        collection_name="pdf_chunks")
    retriever = vector_store.as_retriever(search_type="similarity",
                                        search_kwargs={'k': 8})
    
    # doc_store_page_content = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content") 
    # doc_store_metadata = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/metadata")

    # # Get all azienda names in vector/doc store
    # nome_delle_aziende = set((vector_store.get()['metadatas'][i].get('azienda'), vector_store.get()['metadatas'][i].get('filename')) for i in range(len(vector_store.get()['ids']))) 
    # nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    # azienda_name_records = [x[0] for x in nome_delle_aziende]
    # AZIENDA = azienda_name_records[0]
    
    # Define Re-ranker
    re_ranker = CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512) #HuggingFaceCrossEncoder(model_name=RERANKER_MODEL, model_kwargs={"device": "cpu"})

    # Run Retrieval to get contexts
    logger.info("Retrieving docs...")
    retrieval_output = retrieve(
        retriever = retriever,
        query = QUERY,
        # doc_store_page_content = doc_store_page_content,
        # doc_store_metadata = doc_store_metadata,
        doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
        azienda = AZIENDA,
        pages_joining_str = PAGES_JOINING_STR,
        retrieve_parents = False
    )

    # Run CrossEncode Re-ranker
    if "cross_encode_rerank" in functions_to_run:
        logger.info("Re-Ranking docs via Cross-Encode-Reranker...")
        output = cross_encode_rerank(
            # re_ranker = re_ranker, 
            contexts = retrieval_output.get("context", []),
            question = QUESTION,
            # doc_store_page_content = doc_store_page_content,
            # doc_store_metadata = doc_store_metadata,
            doc_store_large_chunks_path = DOC_STORE_LARGE_CHUNKS_PATH,
            k_min = 2,
            k_max = 5,
            rel_thresh = 0.4,
            max_promoted_parents = 3,
            use_parent_heuristics = False
        )

    if "faster_retrieve_and_rerank" in functions_to_run:
        # Run faster (cross-encode) reranker
        # fast_re_ranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL), top_n=8) # re_ranker compressor for fast retrieve + re_rank+ compression
        fast_reranker_output = faster_retrieve_and_rerank(
            query = QUERY,
            # compressor = fast_re_ranker,
            retriever = retriever,
            azienda = AZIENDA,
            top_n = 4
        )

    # Write output to a file
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write(f"## OUTPUT FOR: re_ranker.py \n{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Save outputs of cross_encode_reranker
        if "cross_encode_rerank" in functions_to_run:
            f.write("# FUNCTION: cross_encode_rerank")
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
        if "faster_retrieve_and_rerank" in functions_to_run:
            f.write(f'\n{"x"*100}\n')
            f.write(f'{"x"*100}\n\n')

            f.write("# FUNCTION: faster_retrieve_and_rerank")
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
    import yaml, os, time, datetime
    from pathlib import Path
    # from langchain_huggingface import HuggingFaceEmbeddings
    from rag_info_extractor.utils.embedder import HFEmbedder
    from langchain_chroma import Chroma
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from rag_info_extractor.utils.common_logging import configure_logging
    import argparse
    from rag_info_extractor.utils.load_config import cfgs

    from rag_info_extractor.rag_pipeline.retrieve import retrieve

    t0 = time.time()

    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Logging for {'-'*30} rag_information_extractor/src/rag_info_extractor/rag_pipeline/retrieve.py")

    # CONFIG FILE SETTINGS:
    cfgs = cfgs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM") 
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    RERANKER_MODEL = cfgs.get("RERANKER_MODEL")
    
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) 
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE) 

    # Query and Aziende (EXAMPLE)
    QUESTION = "Agli amministratori spetta il rimborso delle spese?"
    QUERY = "Agli amministratori spetta il rimborso delle spese?"
    AZIENDA = '2kind srl' # azienda_name_records[0]


    # Run and save outputs of each function
    functions_to_run = ["cross_encode_rerank"] # in ["cross_encode_rerank", "faster_retrieve_and_rerank"]  # can be both
    main(functions_to_run)

    logger.info(f"Total time taken to run the script: {time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))}")