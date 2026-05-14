from typing import Dict, Any, Tuple, List
from pathlib import Path
import json
import sys



# TODO:



def load_company_dicts(combined_raw_json: Path, pred_json: Path, match_scores_json: Path, match_scores_qa_json: Path) -> Tuple[
    Dict[str, Any],  # companies_match_data
    Dict[str, Any],  # companies_match_qa
    Dict[str, Any],  # companies_pred_qa
    Dict[str, Any],  # companies_raw_qa
    Dict[str, Any],  # companies_raw_contexts 
    Dict[str, Any],  # companies_pred_contexts (we save only start and end part not whole context)
    Dict[str, Any],  # companies_raw_contexts_ids
    Dict[str, Any],  # companies_pred_contexts_ids
    Dict[str, Any],  # companies_runtimes
]:
    """
    Read all .json files in combined_raw_json and outputs_dir and build dictionaries keyed by azienda_name.

    Returns tuple of dicts in the order described in the type annotation.

    # combined_raw_json structure:
        {
            "azienda_name": {
                'COMPENSO_DEGLI_AMMINISTRATORI': {
                    "raw_contexts": { 
                        'Rimborso': {
                            'spetta_rimborso': '',
                            'spese_incluse': '',
                        },
                        ...
                    },
                },
                'BILANCI_E_UTILI': {},
                ...
            },  
        }

    # pred_json structure:
        {
            "azienda_name": {
                'COMPENSO_DEGLI_AMMINISTRATORI': {
                    "retrieved_docs": {
                        'Rimborso': {
                            'spetta_rimborso': '',
                            'spese_incluse': '',
                        },
                        ...
                    },
                },
                'BILANCI_E_UTILI': {},
                ...
            },  
        }
    
    # match_scores_json structure:
        {
            "azienda_name": {
                'COMPENSO_DEGLI_AMMINISTRATORI': {
                    'Rimborso': {
                        'spetta_rimborso': 0,
                        'spese_incluse': 0,
                    },
                    ...
                },
                'BILANCI_E_UTILI': {},
                ...
            },
        }

    # match_scores_qa_json structure:
        {
            "azienda_name": {
                'COMPENSO_DEGLI_AMMINISTRATORI': {
                    'Rimborso': 0,
                    ...
                },
                'BILANCI_E_UTILI': {},
                ...
            },
        }


    Args:
        combined_raw_json: Path to dataset directory of json data files
        pred_json: Path to dir where the output.json file is to be saved
    """

    if not combined_raw_json.exists():
        raise FileNotFoundError(f"Dataset json file not found: {combined_raw_json}")
    if not pred_json.exists():
        raise FileNotFoundError(f"Outputs/Predicted json file not found: {pred_json}")
    if not match_scores_json.exists():
        raise FileNotFoundError(f"Match scores json file not found: {match_scores_json}")

    # Initialize output dictionaries
    companies_match_data: Dict[str, Any] = {}
    companies_match_qa: Dict[str, Any] = {}
    companies_pred_qa: Dict[str, Any] = {}
    companies_raw_qa: Dict[str, Any] = {}
    companies_raw_contexts: Dict[str, Any] = {}
    companies_pred_contexts: Dict[str, Any] = {}
    companies_raw_contexts_ids: Dict[str, Any] = {}
    companies_pred_contexts_ids: Dict[str, Any] = {}
    companies_runtimes: Dict[str, Any] = {}

    # json_files_dataset: List[Path] = sorted(combined_raw_json.glob("*.json"))

    # if not json_files_dataset:
    #     print(f"Warning: no .json files found in {combined_raw_json}", file=sys.stderr)
    #     return (
    #         companies_match_data,
    #         companies_pred_qa,
    #         companies_raw_qa,
    #         companies_raw_contexts,
    #         companies_pred_contexts,
    #         companies_raw_contexts_ids,
    #         companies_pred_contexts_ids,
    #         companies_runtimes,
    #     )
    
    # Load raw data from combined_raw_json
    try:
        with combined_raw_json.open("r", encoding="utf-8") as f:
            raw_data = json.load(f) # json tree: azienda -> info_schema -> (values, raw_contexts, raw_contexts_ids, raw_qa)
    except Exception as e:
        print(f"Skipping {combined_raw_json} — failed to read/parse JSON: {e}", file=sys.stderr)

    # get values for raw_qa, raw_contexts from raw data
    for azienda, group in raw_data.items():
        companies_raw_contexts[azienda] = {}
        companies_raw_contexts_ids[azienda] = {}
        companies_raw_qa[azienda] = {} 

        for group_name, group_data in group.items():
            companies_raw_contexts[azienda][group_name] = group_data.get("raw_contexts")
            companies_raw_contexts_ids[azienda][group_name] = group_data.get("raw_contexts_ids")
            companies_raw_qa[azienda][group_name] = group_data.get("raw_qa")


    # # load raw dataset files
    # for jf in json_files_dataset:
    #     try:
    #         with jf.open("r", encoding="utf-8") as f:
    #             data = json.load(f)
    #     except Exception as e:
    #         print(f"Skipping {jf} — failed to read/parse JSON: {e}", file=sys.stderr)
    #         continue

    #     azienda = data.get("azienda_name") or data.get("azienda") or data.get("name")
    #     if not azienda:
    #         print(f"Skipping {jf} — missing 'azienda_name' key", file=sys.stderr)
    #         continue
            
    #     # Safely extract keys with sensible defaults if missing
    #     companies_raw_qa[azienda] = data.get("raw_qa")
    #     companies_raw_contexts[azienda] = data.get("raw_contexts")
    #     companies_raw_contexts_ids[azienda] = data.get("raw_contexts_ids")
    
    # load output/extracted data from pred_json.json 
    try:
        with pred_json.open("r", encoding="utf-8") as f:
            results = json.load(f) # json tree: azienda -> info_schema -> (output, retrieved_docs, re_ranked_docs, rag_qa, run_times, retrieved_docs_texts, re_ranked_docs_texts, optimized_query)
    except Exception as e:
        print(f"Skipping {pred_json} — failed to read/parse JSON: {e}", file=sys.stderr)

    # get the values for outputs/pred_data
    for azienda, res in results.items():
        # per_azienda_match_data = {} # pred_data for each azienda
        per_azienda_contexts = {} # contexts retrieved for each azienda
        per_azienda_contexts_ids = {} # contexts ids retrieved for each azienda
        per_azienda_qa = {} # q & a from rag for each azienda
        per_azienda_runtimes = {} # runtimes of extraction for each azienda

        for mod, v in res.items():
            # per_azienda_match_data[mod] = v.get('match_data') 
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

        # companies_match_data[azienda] = per_azienda_match_data
        companies_pred_qa[azienda] = per_azienda_qa
        companies_pred_contexts[azienda] = per_azienda_contexts
        companies_pred_contexts_ids[azienda] = per_azienda_contexts_ids
        companies_runtimes[azienda] = per_azienda_runtimes
    
    # Load match score data for each azienda from match_scores_json
    try:
        with match_scores_json.open("r", encoding="utf-8") as f:
            companies_match_data = json.load(f)
    except Exception as e:
        print(f"Skipping {match_scores_json} — failed to read/parse JSON: {e}", file=sys.stderr)
        companies_match_data = {}

    # Load match score QA (ragpipeline responses) for each azienda from match_scores_qa_json
    try:
        with match_scores_qa_json.open("r", encoding="utf-8") as f:
            companies_match_qa = json.load(f)
    except Exception as e:
        print(f"Skipping {match_scores_qa_json} — failed to read/parse JSON: {e}", file=sys.stderr)
        companies_match_qa = {}

    
        
    return (
        companies_match_data,
        companies_match_qa,
        companies_pred_qa,
        companies_raw_qa,
        companies_raw_contexts,
        companies_pred_contexts,
        companies_raw_contexts_ids,
        companies_pred_contexts_ids,
        companies_runtimes,
    )




if __name__ == "__main__":

    from rag_info_extractor.utils.load_config import cfgs

    cfgs = cfgs.get("args", {})
    BASE_DIR = Path(cfgs.get("BASE_DIR", ""))

    combined_raw_json = BASE_DIR / "data/jsons/TRAIN/combined_data.json"
    pred_json = BASE_DIR / "runs/TRAIN/custom_chunks/run_2026-03-09 09-11-36/pred.json"
    match_scores_json = BASE_DIR / "runs/TRAIN/custom_chunks/run_2026-03-09 09-11-36/match_scores.json"
    match_scores_qa_json = BASE_DIR / "runs/TRAIN/custom_chunks/run_2026-03-09 09-11-36/match_scores_qa.json"
    
    data = load_company_dicts(
        combined_raw_json=combined_raw_json,
        pred_json=pred_json,
        match_scores_json=match_scores_json,
        match_scores_qa_json=match_scores_qa_json
    )

    idx = 7
    print(json.dumps(data[idx], indent=4, ensure_ascii=False))
    print(data[idx]['2kind srl']['BILANCI_E_UTILI'].keys())
