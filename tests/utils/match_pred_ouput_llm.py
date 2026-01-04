from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# python native
from typing import Dict, Any, List, Tuple, Optional
import json
import copy
import re
import yaml
from pathlib import Path
import sys

# Logging
import logging
logger = logging.getLogger(__name__)

# ---- MATCH TEMPLATE ----
MATCH_TEMPLATE = {
    'CompensoAmministratori': {
        'Rimborso': {
            'spetta_rimborso': 0,
            'spese_incluse': 0,
        },
        'IndennitaAnnuale': {
            'spetta_indennita_da_soci': 0,
            'misura_indennita': 0
        },
        'IndennitaCessazione': {
            'spetta_indennita': 0
        },
        # 'Context': 0,
    },
    'BilanciUtili': {
        'PercentualeRiservaLegale': {
            'percentuale_utili': 0
        },
        'CapitaleSociale': {
            'capitale_sociale_euro': 0
        },
        'TermineApprovazioneBilancio': {
            'termine_ordinario_giorni': 0,
            'termine_prorogato_giorni': 0
        },
        'DataChiusuraEsercizio': {
            'data_chiusura_esercizio': 0
        },
        'UtiliResidui': {
            'utili_residui': 0
        },
        # 'Context': 0,
    },
    'InfoGenerali': {
        "Durata": {
            "durata_dell_azienda": 0
        },
        # 'Context': 0,
    },
}

# ======================================
# 2) Utilities for walking nested dicts
# ======================================
def leaf_paths(template: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """Return all leaf key paths (exclude 'Context' counters)."""
    out: List[Tuple[str, ...]] = []

    def _walk(node: Any, path: Tuple[str, ...]):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "Context":
                    continue
                _walk(v, path + (k,))
        else:
            out.append(path)

    _walk(template, tuple())
    return out

def set_by_path(d: Dict[str, Any], path: Tuple[str, ...], val: Any) -> None:
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = val

def get_by_path(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
        if cur is None:
            return None
    return cur

def compute_context_sums(match_dict: Dict[str, Any], present_fields: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Optional: set section 'Context' as the sum of field presences you provide.
    Example present_fields:
      {
        "CompensoAmministratori": ["spetta_rimborso","spese_incluse","spetta_indennita_da_soci","misura_indennita","spetta_indennita"],
        "BilanciUtili": ["percentuale_utili","capitale_sociale_euro","termine_ordinario_giorni","termine_prorogato_giorni","data_chiusura_esercizio","utili_residui"],
        "InfoGenerali": ["durata_dell_azienda"]
      }
    """
    if not present_fields:
        return
    for section, fields in present_fields.items():
        match_dict[section]["Context"] = len(fields)


# ==========================================
# 3) Lightweight value normalization helpers
# ==========================================
_BOOL_TRUE = {"true", "yes", "sì", "si", "y", "vero"}
_BOOL_FALSE = {"false", "no", "n", "falso"}

def normalize_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()

    # Normalize numbers with punctuation (e.g., 1.000,00 vs 1000.00)
    s_num = re.sub(r"[,\s]", "", s)
    if re.fullmatch(r"-?\d+(\.\d+)?", s_num):
        try:
            # prefer int when possible
            f = float(s_num)
            return int(f) if f.is_integer() else f
        except Exception:
            pass

    # Normalize bool-ish strings
    sl = s.lower()
    if sl in _BOOL_TRUE:
        return True
    if sl in _BOOL_FALSE:
        return False

    # Normalize date formats lightly (yyyy-mm-dd / dd-mm-yyyy etc.) -> just digits
    digits = re.sub(r"\D", "", s)
    if 6 <= len(digits) <= 8:  # a very rough date signal
        return digits

    return s.lower()  # case-insensitive compare for text

def fast_equal(a: Any, b: Any) -> bool:
    return normalize_scalar(a) == normalize_scalar(b)


# =======================================
# 4) Pydantic structured output (per key)
# =======================================
class FieldDecision(BaseModel):
    match: bool = Field(..., description="True se PREDICTED uguale REFERENCE semanticamente; altrimeni False.")

EVAL_SYSTEM = (
    "Sei un valutatore severo ma equo per un singolo campo.\n"
    "Stabilisci se PREDICTED corrisponde semanticamente a REFERENCE:\n"
    "- Accetta differenze banali di formattazione (1,000 vs 1000; 31/12/2024 vs 2024-12-31).\n"
    "- yes/true/sì → True; no/false → False.\n"
    "- Per numeri/percentuali/durate, confronta il valore sottostante.\n"
    "- Se uno dei due valori è mancante mentre l’altro è presente, è una mancata corrispondenza.\n"
    "{format_instructions}"
)

EVAL_USER = (
    "Field: {field_path}\n"
    "REFERENCE: {reference}\n"
    "PREDICTED: {predicted}\n"
    "Return your decision."
)

def build_field_chain(llm) -> Any:
    """
    Build a LangChain chain: Prompt -> LLM -> Pydantic parser.
    """
    parser = PydanticOutputParser(pydantic_object=FieldDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EVAL_SYSTEM),
        ("user", EVAL_USER),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser



# =======================================================
# 5) Main function: field-by-field with one LLM call each
# =======================================================
def evaluate_pred_fieldwise(
    reference_obj: Dict[str, Any],
    predicted_obj: Dict[str, Any],
    present_fields: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Returns a filled match_data dict with 1/0 per leaf.
    One LLM call per field (skips call if fast_equal == True).
    """
    match_dict = copy.deepcopy(MATCH_TEMPLATE)
    
    llm = ChatOllama(
        model=EVALUATOR_LLM,
        temperature=0,
        num_predict=64,   # short outputs
        format="json",
        cache=False    
    )
    
    chain = build_field_chain(llm)

    for path in leaf_paths(match_dict):
        print()###
        ref_v = get_by_path(reference_obj, path)
        pred_v = get_by_path(predicted_obj, path)

        print(f"Node: {path}")###
        print(f"ref_v: {ref_v}")###
        print(f"pred_v: {pred_v}")###

        # Fast path: exact/normalized equality -> no LLM call
        if fast_equal(ref_v, pred_v):
            print("Fast Path...")###
            set_by_path(match_dict, path, 1)
            continue
        
        # If pred_v empty or ref_v empty while other is not then set it to 0 -> no LLM call (since otherwise prev. if-statement will be executed)
        if not ref_v:
            print("NOT MATCHED...")
            set_by_path(match_dict, path, 0)
            continue        
        if not pred_v:
            print("NOT MATCHED...")
            set_by_path(match_dict, path, 0)
            continue
    

        # LLM decision (single field)
        payload = {
            "field_path": " / ".join(path),
            "reference": json.dumps(ref_v, ensure_ascii=False) if isinstance(ref_v, (dict, list)) else ref_v,
            "predicted": json.dumps(pred_v, ensure_ascii=False) if isinstance(pred_v, (dict, list)) else pred_v,
        }

        try:
            decision: FieldDecision = chain.invoke(payload)
            print(decision)###
            set_by_path(match_dict, path, 1 if decision.match else 0)
        except Exception as e:
            # If parsing fails, be conservative
            print(f"WARNING!: Exception: {e}")
            set_by_path(match_dict, path, 0)

    # Optional Context counters
    compute_context_sums(match_dict, present_fields)
    return match_dict



if __name__ == "__main__":
    # CONFIG FILE SETTINGS  (Load args form config file)
    cfg_path = Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    cfgs = configs.get("args", {})
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM", "")

    args_inserted = sys.argv
    assert len(args_inserted) == 3, "Usage: python <current filename> <full path to raw_data_json> <full path to pred_data_json>"

    raw_data = json.loads(args_inserted[1], indent=4, ensure_ascii=False)
    pred_data = json.loads(args_inserted[2], indent=4, ensure_ascii=False)
    
    # Match and score pred vs raw and save the results
    logger.info("Evaluating Predicted output w.r.t Raw output using LLM as a judge.")
    match_output = evaluate_pred_fieldwise(raw_data, pred_data)
    with open("../results/last_run_match_pred_output_llm.json", "w") as f:
        json.dump(match_output, f, indent=4, ensure_ascii=False)
    
    logger.info("Evaluation completed. Results saved to: \t '../results/last_run_match_pred_output_llm.json'")