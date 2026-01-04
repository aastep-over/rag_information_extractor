from langchain_core.documents import Document

# python native
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

# -------- Obtain chunk IDs for contexts -------------------
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


def find_raw_chunk_ids(
    raw_contexts: Dict[str, Dict[str, str]],
    azienda_name: str,
    parent_chunks: List[Document],
    children_chunks: List[Document]
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    
    """Calculate id of the chunk in vector/doc store containing the reference context"""
    
    raw_contexts_ids = {}
    for group_name, group in raw_contexts.items():
        per_group = {}
        for sg_name, context in group.items():
            per_subgruop = {}
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