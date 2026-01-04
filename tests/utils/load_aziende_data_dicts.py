from typing import Dict, Any, Tuple, List
from pathlib import Path
import json
import sys


def load_company_dicts(dataset_dir: Path, json_files_output: Path) -> Tuple[
    Dict[str, Any],  # companies_match_data
    Dict[str, Any],  # companies_pred_qa
    Dict[str, Any],  # companies_raw_qa
    Dict[str, Any],  # companies_raw_contexts
    Dict[str, Any],  # companies_pred_contexts
    Dict[str, Any],  # companies_raw_contexts_ids
    Dict[str, Any],  # companies_pred_contexts_ids
    Dict[str, Any],  # companies_runtimes
]:
    """
    Read all .json files in dataset_dir and outputs_dir and build dictionaries keyed by azienda_name.

    Returns tuple of dicts in the order described in the type annotation.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not json_files_output.exists():
        raise FileNotFoundError(f"Outputs File not found: {json_files_output}")

    # Initialize output dictionaries
    companies_match_data: Dict[str, Any] = {}
    companies_pred_qa: Dict[str, Any] = {}
    companies_raw_qa: Dict[str, Any] = {}
    companies_raw_contexts: Dict[str, Any] = {}
    companies_pred_contexts: Dict[str, Any] = {}
    companies_raw_contexts_ids: Dict[str, Any] = {}
    companies_pred_contexts_ids: Dict[str, Any] = {}
    companies_runtimes: Dict[str, Any] = {}

    json_files_dataset: List[Path] = sorted(dataset_dir.glob("*.json"))

    if not json_files_dataset:
        print(f"Warning: no .json files found in {dataset_dir}", file=sys.stderr)
        return (
            companies_match_data,
            companies_pred_qa,
            companies_raw_qa,
            companies_raw_contexts,
            companies_pred_contexts,
            companies_raw_contexts_ids,
            companies_pred_contexts_ids,
            companies_runtimes,
        )
    

    # load raw dataset files
    for jf in json_files_dataset:
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {jf} — failed to read/parse JSON: {e}", file=sys.stderr)
            continue

        azienda = data.get("azienda_name") or data.get("azienda") or data.get("name")
        if not azienda:
            print(f"Skipping {jf} — missing 'azienda_name' key", file=sys.stderr)
            continue
            
        # Safely extract keys with sensible defaults if missing
        companies_raw_qa[azienda] = data.get("raw_qa")
        companies_raw_contexts[azienda] = data.get("raw_contexts")
        companies_raw_contexts_ids[azienda] = data.get("raw_contexts_ids")
    
    # load preds from output.json 
    try:
        with json_files_output.open("r", encoding="utf-8") as f:
            results = json.load(f) # json tree: azienda -> info_schema -> (output, retrieved_docs, re_ranked_docs, rag_qa, run_times, retrieved_docs_texts, re_ranked_docs_texts, optimized_query)
    except Exception as e:
        print(f"Skipping {json_files_output} — failed to read/parse JSON: {e}", file=sys.stderr)

    # get the values for outputs/pred_data
    for azienda, res in results.items():
        per_azienda_match_data = {} # pred_data for each azienda
        per_azienda_contexts = {} # contexts retrieved for each azienda
        per_azienda_contexts_ids = {} # contexts ids retrieved for each azienda
        per_azienda_qa = {} # q & a from rag for each azienda
        per_azienda_runtimes = {} # runtimes of extraction for each azienda

        for mod, v in res.items():
            per_azienda_match_data[mod] = v.get('match_data') 
            per_azienda_contexts[mod] = { 
                'retrieved_docs_texts': v.get('retrieved_docs_texts'),
                're_ranked_docs_texts': v.get('re_ranked_docs_texts')
            }
            per_azienda_contexts_ids[mod] = { 
                'retrieved_docs': v.get('retrieved_docs'),
                're_ranked_docs': v.get('re_ranked_docs'),
            }
            per_azienda_qa[mod] = v.get('rag_qa')
            per_azienda_runtimes[mod] = v.get('run_times')

        companies_match_data[azienda] = per_azienda_match_data
        companies_pred_qa[azienda] = per_azienda_qa
        companies_pred_contexts[azienda] = per_azienda_contexts
        companies_pred_contexts_ids[azienda] = per_azienda_contexts_ids
        companies_runtimes[azienda] = per_azienda_runtimes
        
    return (
        companies_match_data,
        companies_pred_qa,
        companies_raw_qa,
        companies_raw_contexts,
        companies_pred_contexts,
        companies_raw_contexts_ids,
        companies_pred_contexts_ids,
        companies_runtimes,
    )




if __name__ == "__main__":
    print(load_company_dicts(
        dataset_dir=Path("../data/temp"),
        json_files_output=Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/outputs/temp/custom_chunks/output.json")
    ))