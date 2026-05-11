from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# python native
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# -------- HELPERS for finding matching chunks/contexts -------------------
def _normalize(s: str) -> str:
    # Light normalization helps SequenceMatcher:
    # - lowercase
    # - collapse whitespace
    # - strip leading/trailing spaces
    return re.sub(r"\s+", " ", s.lower()).strip()

def _coverage_ratio(ref: str, chunk: str) -> float:
    """
    Coverage(ref, chunk) = total matched characters between ref and chunk / len(ref)
    using difflib.SequenceMatcher. This captures 'how much of REF is present in CHUNK'.
    """
    if not ref:
        return 0.0
    sm = SequenceMatcher(None, ref, chunk, autojunk=False)
    match_len = sum(block.size for block in sm.get_matching_blocks() if block.size > 0)
    return match_len / len(ref)

def find_covering_chunk_indices_difflib(
    reference_contexts: List[str],
    chunks: List[Dict[str, Any]],
    threshold: float = 0.7,
    normalize: bool = True,
) -> List[int]:
    """
    For each reference context, find the index of the chunk that covers at least
    `threshold` fraction of its characters using difflib.SequenceMatcher coverage.
    If none qualifies, return -1 for that reference.

    Args:
        reference_contexts: list of ground-truth passages.
        chunks: list of retrieved chunk strings.
        threshold: minimum coverage required (0..1).
        normalize: whether to apply simple normalization before comparison.

    Returns:
        List[int]: one index per reference context, or -1 if no chunk meets threshold.
    """
    if normalize:
        refs = [_normalize(r or "") for r in reference_contexts]
        chs  = [_normalize(c['text'] or "") for c in chunks]
    else:
        refs = [r or "" for r in reference_contexts]
        chs  = [c['text'] or "" for c in chunks]

    indices: List[int] = []
    for ref in refs:
        if not ref:
            indices.append(-1)
            continue

        best_idx: Optional[int] = None
        best_cov: float = 0.0

        for i, ch in enumerate(chs):
            cov = _coverage_ratio(ref, ch)
            if cov > best_cov:
                best_cov = cov
                best_idx = i

        indices.append(best_idx if (best_idx is not None and best_cov >= threshold) else -1)

    return [chunks[i].get("id") for i in indices if i != -1] #type: ignore 


def format_context_ids_lists_in_json(context_ids: Dict[str, Dict[str, Dict[str, List[int]]]]) -> str:
    """
    Formats a dictionary to a JSON string, keeping dicts indented 
    but collapsing integer lists onto a single line.
    """
    json_str = json.dumps(context_ids, indent=4)

    # helper function to format the contents of the matched lists
    def _collapse_list(match):
        # Extract the contents, removing all whitespace and newlines
        compressed = re.sub(r'\s+', '', match.group(1))
        # Re-add a single space after commas for standard readability (e.g., [13, 14])
        compressed = compressed.replace(',', ', ')
        return f"[{compressed}]"
    
    formatted_json_str = re.sub(r'\[([-\d\s,]*)\]', _collapse_list, json_str)

    return formatted_json_str



# ----------- Functions to load chunks from doc_store and vector_db ------------------
def load_parent_chunks_from_dir(chunks_dir: str) -> List[Document]:
    """ Load parent chunks from txt/binary files stored in chunks_dir"""
    
    parent_chunks = [Document(page_content="", metadata={}) for i in range(len(os.listdir(f"{chunks_dir}/page_content")))] 
    content_files = os.listdir(f"{chunks_dir}/page_content")
    metadata_files = os.listdir(f"{chunks_dir}/metadata")
    
    for i, (content_file, metadata_file) in enumerate(zip(content_files, metadata_files)):
        with open(f"{chunks_dir}/page_content/{content_file}", encoding="utf-8") as f:
            parent_chunks[i].page_content = f.read()
        with open(f"{chunks_dir}/metadata/{metadata_file}", encoding="utf-8") as f:
            parent_chunks[i].metadata = json.load(f)

    return parent_chunks

def load_children_chunks_from_chroma(chunks_dir: str, embedding) -> List[Document]:    
    """ Load children chunks from a vector db (chroma) in chunks_dir"""
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=chunks_dir,
        collection_name="pdf_chunks"
    )
    children_chunks = [Document(page_content="", metadata={}) for i in range(len(vector_store.get()['ids']))]
    for i, (content, metadata) in enumerate(zip(vector_store.get()['documents'], vector_store.get()['metadatas'])):
        children_chunks[i].page_content = content
        children_chunks[i].metadata = metadata
    
    return children_chunks
        

# ---------------------- MAIN ---------------------------------------------------
def find_raw_chunk_ids(
    raw_contexts: Dict[str, Dict[str, Dict[str, str]]],
    azienda_name: str,
    parent_chunks: List[Document],
    children_chunks: List[Document]
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    
    """Calculate id of the chunk in vector/doc store containing the reference context"""
    
    raw_contexts_ids = {}
    for group_name, group in raw_contexts.items():
        group_contexts = group.get("raw_contexts", {}) # obtain raw_contexts
        per_group = {}
        for sg_name, context in group_contexts.items():
            per_subgruop = {}
            
            # if not context: # TODO: REMOVE THIS PART AFTER VERIFYING CONTEXT IDS
            #     per_subgruop["parents"] = [-100]
            #     per_subgruop["children"] = [-100]
            #     per_group[sg_name] = per_subgruop
            #     continue

            
            # find ids of children chunks
            rel_chunks_children = [{"text": d.page_content, "id": d.metadata.get("chunk_id")} for d in children_chunks if d.metadata.get("azienda") == azienda_name.lower()]
            rel_ids_children = find_covering_chunk_indices_difflib([context], rel_chunks_children)

            # find ids of parent chunks
            rel_chunks_parents = [{"text": d.page_content, "id": d.metadata.get("chunk_id")} for d in parent_chunks if d.metadata.get("azienda") == azienda_name.lower()]
            rel_ids_parents = find_covering_chunk_indices_difflib([context], rel_chunks_parents)
           

            per_subgruop["parents"] = rel_ids_parents
            per_subgruop["children"] = rel_ids_children
            
            per_group[sg_name] = per_subgruop
        
        raw_contexts_ids[group_name] = per_group
    
    return raw_contexts_ids


def insert_chunk_id_to_combined_raw_json(
    combined_raw_json_path: Path,
    parent_chunks: List[Document],
    children_chunks: List[Document]
):
    """
    Find chunk ids for raw contexts (for both parents(doc_store) and children(vector_db) chunks) 
    and insert them in combined_raw_json.

    Args:
        combined_raw_json_path: Path to combined_raw_json file.
        parent_chunks: List of parent chunks.
        children_chunks: List of children chunks.
    """
    combined_raw_data = json.loads(combined_raw_json_path.read_text(encoding="utf-8"))

    for azienda, data in combined_raw_data.items():
        # Find raw chunks
        logger.info(f"Finding raw chunks for azienda: {azienda}")
        raw_contxts_ids = find_raw_chunk_ids(
            raw_contexts = data,
            azienda_name = azienda,
            parent_chunks = parent_chunks,
            children_chunks = children_chunks
        )
        
        # update raw_context_ids in combined_raw_json
        for group, sub_group in raw_contxts_ids.items():
            for sub_group_name, context_ids in sub_group.items():
                data[group]["raw_contexts_ids"][sub_group_name] = context_ids
    
    # Save updated json
    combined_raw_json_path.write_text(json.dumps(combined_raw_data, indent=4, ensure_ascii=False), encoding="utf-8")
        


# -------------------- Function to verify if obtained ids correct --------------------------------------
def analyze_raw_contexts_coverage(
    combined_raw_json_path: str,
    parent_chunks_dir: str,
    children_chunks: List[Document],
    threshold: float = 0.7,
) -> dict:
    """
    Loads the combined_raw_json, reads parent and child chunks, and checks coverage of raw_contexts.
    Outputs entries for which coverage is below the given threshold.
    """
    import json
    import os

    # Helper normalization function (adapt to match _normalize logic)
    def _normalize(text: str) -> str:
        text = text or ""
        return ' '.join(text.strip().lower().split())

    # Helper: Read a single parent chunk by id (=chunk_id)
    def _read_parent_chunk(chunk_id: int) -> str:
        try:
            with open(f"{parent_chunks_dir}/page_content/{chunk_id}", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    # Helper: Read a single child chunk by id (=chunk_id)
    def _read_child_chunk(chunk_id: int) -> str:
        # children_chunks is already sorted
        return children_chunks[chunk_id].page_content


    # Coverage metric: fraction of raw_context normalized text contained over total length
    def _context_coverage(reference: str, chunk: str) -> float:
        if not reference: return 0.0
        # Use difflib.SequenceMatcher for more robust matching
        import difflib
        sm = difflib.SequenceMatcher(None, reference, chunk)
        matches = sm.get_matching_blocks()
        overlap = sum(m.size for m in matches)
        return overlap / max(1, len(reference))

    # sort children_chunks by chunk_id
    children_chunks.sort(key=lambda x: x.metadata.get("chunk_id", ""))


    # Load the combined_raw_json
    with open(combined_raw_json_path, encoding="utf-8") as f:
        combined_raw_json = json.load(f)

    results_below_threshold = {}

    for azienda, data in combined_raw_json.items():
        for group_key, group in data.items():
            raw_contexts_per_group = group.get("raw_contexts", {})
            raw_contexts_ids_per_group = group.get("raw_contexts_ids", {})

            for sub_key in raw_contexts_per_group.keys(): # raw_contexts and raw_contexts_ids have same subkeys (subgroup names)
                raw_context = raw_contexts_per_group[sub_key] # context (str)
                raw_context_ids = raw_contexts_ids_per_group[sub_key] # {"parent": [], "children: []"}

                if not raw_context or not raw_context_ids:
                    continue
                
                ref_context = raw_context # _normalize(raw_context)
                parent_ids = raw_context_ids.get("parents", [])
                child_ids = raw_context_ids.get("children", [])

                # Load and normalize parent/child chunk texts
                parent_chunks = [_read_parent_chunk(pid) for pid in parent_ids if pid is not None] # _normalize(_read_parent_chunk(pid))
                child_chunks = [_read_child_chunk(cid) for cid in child_ids if cid is not None] # _normalize(_read_child_chunk(cid))

                # Check coverage of ref_context in each parent and child chunk
                parent_coverage = _context_coverage(ref_context, " ".join(parent_chunks))
                child_coverage = _context_coverage(ref_context, " ".join(child_chunks))


                # Collect entries below threshold
                if (parent_coverage < threshold) or (child_coverage < threshold): 
                    if azienda not in results_below_threshold:
                        results_below_threshold[azienda] = {}
                    if group_key not in results_below_threshold[azienda]:
                        results_below_threshold[azienda][group_key] = {}
                    if sub_key not in results_below_threshold[azienda][group_key]:
                        results_below_threshold[azienda][group_key][sub_key] = {}

                    results_below_threshold[azienda][group_key][sub_key] = {
                        "raw_context": raw_context,
                        "parent_ids": parent_ids,
                        "child_ids": child_ids,
                        "parent_coverage": parent_coverage,
                        "child_coverage": child_coverage,
                    }

    return results_below_threshold


if __name__ == "__main__":

    from rag_info_extractor.utils.load_config import cfgs
    import time, datetime
    import argparse
    from dotenv import load_dotenv

    from rag_info_extractor.utils.common_logging import configure_logging
    from rag_info_extractor.utils.embedder import HFEmbedder
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-type",
        type=str,
        choices=["custom_chunks", "fixed_size_chunks", "semantic_chunks", "custom_chunks_2"],
        help="Chunking method used to extract context from",
        required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["TEST", "TRAIN"],
        required=True
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Logging for {"-"*30} tests/utils/find_raw_chunk_ids.py")

    t0 = time.time()

    # CONFIG FILE SETTINGS:
    cfgs = cfgs.get("args", {})
    BASE_DIR = cfgs.get("BASE_DIR", "./")

    # Obtain raw_data dict
    combined_raw_json_path = Path(BASE_DIR, f"data/jsons/{args.dataset}/{args.chunk_type}/combined_data.json") 

    # Obtain larger chunks (parent chunks) as Document
    DOC_STORE_LARGE_CHUNKS_PATH = f"{BASE_DIR}/data/large_chunks_dbs/{args.dataset}/{args.chunk_type}"
    parent_chunks = load_parent_chunks_from_dir(f"{DOC_STORE_LARGE_CHUNKS_PATH}")
    logger.info(f"Loaded parent {len(parent_chunks)} chunks from: {DOC_STORE_LARGE_CHUNKS_PATH}")

    # Obtain child/small chunks from vector db (chroma)
    embedding = HFEmbedder(normalize_embeddings=True)
    VECTOR_STORE_PATH = f"{BASE_DIR}/data/vector_dbs/{args.dataset}/{args.chunk_type}"
    children_chunks = load_children_chunks_from_chroma(VECTOR_STORE_PATH, embedding)
    logger.info(f"Loaded {len(children_chunks)} children chunks from: {VECTOR_STORE_PATH}")


    # # 1. raw chunk ids for a single azienda
    # azienda = "medicare salute & servizi s.r.l."
    # logger.info(f"Finding raw chunks ids for a single azienda...'{azienda}'")
    # combined_raw_data = json.loads(combined_raw_json_path.read_text(encoding="utf-8"))
    # data_azienda = combined_raw_data.get(azienda, {})
    # raw_contxts_ids = find_raw_chunk_ids(
    #         raw_contexts = data_azienda,
    #         azienda_name = azienda,
    #         parent_chunks = parent_chunks,
    #         children_chunks = children_chunks
    #     )
    # raw_contxts_ids = format_context_ids_lists_in_json(raw_contxts_ids)
    # print(raw_contxts_ids)


    # # 2. Find raw chunk ids for all aziende and update combined_raw_data.json file in data/jsons/
    # logger.info("Finding raw chunks ids...")
    # insert_chunk_id_to_combined_raw_json(
    #     combined_raw_json_path = combined_raw_json_path,
    #     parent_chunks = parent_chunks,
    #     children_chunks = children_chunks
    # )
    # logger.info(f"Found raw chunks and updated {combined_raw_json_path}")

    # 3. Verify if raw chunk ids correctly found
    results_below_threshold = analyze_raw_contexts_coverage(
        combined_raw_json_path = str(combined_raw_json_path),
        parent_chunks_dir = DOC_STORE_LARGE_CHUNKS_PATH,
        children_chunks = children_chunks,
        threshold = 0.7
    )

    with open("output_temp.json", "w", encoding="utf-8") as f:
        f.write("Output for find_raw_chunks_ids.py\n")
        f.write(f"Date: {datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Function ran: analyze_raw_contexts_coverage\n\n\n")

        json.dump(results_below_threshold, f, indent=4, ensure_ascii=False)
    print(json.dumps(results_below_threshold, indent=4, ensure_ascii=False))


    logger.info(f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}")






# Workflow for Verifying raw_chunks:
# 1. Read output_temp.json saved byfind_raw_chunks_id.py to check which fileds have been outputed with low coverage threshold score
# 2. Read the chunk of the idxs for which cov_threshold output is low in DEMO.ipynb (for parent and/or child)
# 3. Compare and update combined_data.json if needed


## Verified for TRAIN/custom_chunks