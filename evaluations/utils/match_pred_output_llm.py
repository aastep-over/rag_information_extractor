import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# GEMINI API 
from google import genai
from google.api_core import exceptions  # 429 RESOURCE_EXHAUSTED.
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential

logger = logging.getLogger(__name__)

# ---- MATCH TEMPLATE ----
MATCH_TEMPLATE = {
    "COMPENSO_DEGLI_AMMINISTRATORI": {
        "Rimborso": {
            "spetta_rimborso": 0,
            "spese_incluse": 0,
        },
        "IndennitaAnnuale": {"spetta_indennita_da_soci": 0, "misura_indennita": 0},
        "IndennitaCessazione": {"spetta_indennita": 0},
    },
    "BILANCI_E_UTILI": {
        "PercentualeRiservaLegale": {"percentuale_utili": 0},
        "CapitaleSociale": {"capitale_sociale_euro": 0},
        "TermineApprovazioneBilancio": {
            "termine_ordinario_giorni": 0,
            "termine_prorogato_giorni": 0,
        },
        "DataChiusuraEsercizio": {"data_chiusura_esercizio": 0},
        "UtiliResidui": {"utili_residui": 0},
        # 'Context': 0,
    },
    "INFO_GENERALI": {
        "Durata": {"durata_dell_azienda": 0},
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


def compute_context_sums(
    match_dict: Dict[str, Any], present_fields: Optional[Dict[str, List[str]]] = None
) -> None:
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


def convert_combined_json_to_match_template(
    combined_json: Dict[str, Any], data_type: Literal["raw", "pred"]
):
    """
    Converts the following dicts to --> MATCH TEMPLATE: (removes the keys: values and output)

    combined_raw_json:
        {
            'COMPENSO_DEGLI_AMMINISTRATORI': {
                "values": {
                    'Rimborso': {
                        'spetta_rimborso': '',
                        'spese_incluse': '',
                    },
                    ...
            },
            'BILANCI_E_UTILI': {},
            ...
        }

    combined_pred_json:
        {
            'COMPENSO_DEGLI_AMMINISTRATORI': {
                "output": {
                    'Rimborso': {
                        'spetta_rimborso': '',
                        'spese_incluse': '',
                    },
                    ...
            },
            'BILANCI_E_UTILI': {},
            ...
        }
    """
    match_template = {}
    for group_name, group in combined_json.items():
        match_template[group_name] = {}
        if data_type == "raw":
            group_data = group.get("values", {})
        elif data_type == "pred":
            group_data = group.get("output", {})

        match_template[group_name] = group_data

    return match_template


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
    match: bool = Field(
        ...,
        description="True se PREDICTED uguale REFERENCE semanticamente; altrimeni False.",
    )


EVAL_SYSTEM = (
    "Sei un valutatore severo ma equo per un singolo campo.\n"
    "Stabilisci se PREDICTED corrisponde semanticamente a REFERENCE:\n"
    "- Accetta differenze banali di formattazione (1,000 vs 1000; 31/12/2024 vs 2024-12-31; 31 dicembre vs 31 dicembre di ogni anno/anno solare; Indeterminata/o vs tempo indeterminato; 10% vs 10 vs dieci per cento).\n"
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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVAL_SYSTEM),
            ("user", EVAL_USER),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


# =======================================
# 4.5) Evaluation with GOOGLE API
# =======================================
@retry(wait=wait_random_exponential(min=1, max=60))
def evaluate_with_GEMINI(
    path: Tuple[str, ...],
    reference: str,
    predicted: str,
    system_prompt: str,
    user_prompt: str,
):

    user_prompt = user_prompt.replace("{field_path}", " / ".join(path))
    user_prompt = user_prompt.replace("{reference}", reference)
    user_prompt = user_prompt.replace("{predicted}", predicted)

    content = f"SYSTEM:\n{system_prompt} \n\nUSER:\n{user_prompt}"

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = client.models.generate_content(model="gemma-3-27b-it", contents=content)

    match_true = re.match(r"true.*", response.text.strip(), re.IGNORECASE)  # type: ignore
    match_false = re.match(r"false.*", response.text.strip(), re.IGNORECASE)  # type: ignore

    if match_true:
        return True
    if match_false:
        return False

    return "N/A"


# =======================================================
# 5) Main function: field-by-field with one LLM call each
# =======================================================
def evaluate_pred_fieldwise(
    reference_obj: Dict[str, Any],
    predicted_obj: Dict[str, Any],
    log_file: Path,
    local_evaluator_llm: str,
    present_fields: Optional[Dict[str, List[str]]] = None,
    use_gemini: bool = False,
) -> Dict[str, Any]:
    """
    Returns a filled match_data dict with 1/0 per leaf.
    One LLM call per field (skips call if fast_equal == True).
    """
    match_dict = copy.deepcopy(MATCH_TEMPLATE)

    # Covert reference_ob and predicted_obj to match with match_dict keys
    reference_obj = convert_combined_json_to_match_template(reference_obj, "raw")
    predicted_obj = convert_combined_json_to_match_template(predicted_obj, "pred")

    llm = ChatOllama(
        model=local_evaluator_llm,
        temperature=0,
        num_predict=64,  # short outputs
        format="json",
        cache=False,
    )

    chain = build_field_chain(llm)

    with open(log_file, "a", encoding="utf-8") as f:

        for path in leaf_paths(match_dict):
            ref_v = get_by_path(reference_obj, path)
            pred_v = get_by_path(predicted_obj, path)

            f.write(f"\nNode: {path}\n")
            f.write(f"ref_v: {ref_v}\n")
            f.write(f"pred_v: {pred_v}\n")

            print(f"Node: {path}")  ###
            print(f"ref_v: {ref_v}")  ###
            print(f"pred_v: {pred_v}")  ###

            # Fast path: exact/normalized equality -> no LLM call
            if fast_equal(ref_v, pred_v):
                f.write("Fast Path...\n")
                print("Fast Path...")  ###
                set_by_path(match_dict, path, 1)
                continue

            # Fast path 2: if ref_v is empty and pred_v says something relevant to missing/not found
            not_found_pattern = r"\b(?:non\s+(?:specificat[oa]|indicat[oa]|definit[oa]|dichiarat[oa]|riportat[oa]|disponibil[ea]|menzionat[oa]|fornit[oa]|trovat[oa]|individuat[oa]|reperit[oa]|riscontrat[oa]|rinvenut[oa])|nessun[oa]?\s+(?:risultat[oa]|rispost[ae]?|riscontr[oi]|element[oi]|dat[oi])|assenza\s+di\s+(?:risultat[oi]|rispost[ae]?|riscontr[oi]|informazion[ei])|informazion[ei]\s+(?:mancant[ei]|indisponibil[ei]|assent[ei])|dat[oi]\s+mancant[ei]|ricerca\s+(?:infruttuos[ao]|senza\s+esito|priva\s+di\s+esiti)|risposta\s+non\s+pervenuta)\b"
            if not ref_v and re.search(not_found_pattern, pred_v, re.IGNORECASE):
                f.write("Fast Path 2...\n")
                print("Fast Path 2...")  ###
                set_by_path(match_dict, path, 1)
                continue

            # If pred_v empty or ref_v empty while other is not then set it to 0 -> no LLM call (since otherwise prev. if-statement will be executed)
            if not ref_v:
                f.write("NOT MATCHED...\n")
                print("NOT MATCHED...")
                set_by_path(match_dict, path, 0)
                continue
            if not pred_v:
                f.write("NOT MATCHED...\n")
                print("NOT MATCHED...")
                set_by_path(match_dict, path, 0)
                continue

            if use_gemini:
                try:
                    decision_gemini = evaluate_with_GEMINI(
                        path=path,
                        reference=ref_v,
                        predicted=pred_v,
                        system_prompt=EVAL_SYSTEM,
                        user_prompt=EVAL_USER,
                    )
                    decision: FieldDecision = FieldDecision(match=decision_gemini) if decision_gemini != "N/A" else FieldDecision(match=False)  # type: ignore
                except Exception as e:
                    # If parsing fails, be conservative
                    f.write(f"WARNING!: Exception: {e}")
                    print(f"WARNING!: Exception: {e}")
                    # close file and break code when limit hits
                    f.close()
                    raise Exception(e)
                    # set_by_path(match_dict, path, 0)
                else:
                    f.write(f"Decision: {decision_gemini}")
                    print(decision_gemini)  ###
                    set_by_path(match_dict, path, 1 if decision.match else 0)
            else:
                # LLM decision (single field)
                payload = {
                    "field_path": " / ".join(path),
                    "reference": (
                        json.dumps(ref_v, ensure_ascii=False)
                        if isinstance(ref_v, (dict, list))
                        else ref_v
                    ),
                    "predicted": (
                        json.dumps(pred_v, ensure_ascii=False)
                        if isinstance(pred_v, (dict, list))
                        else pred_v
                    ),
                }

                try:
                    decision: FieldDecision = chain.invoke(payload)
                    f.write(f"Decision: {decision}")
                    print(decision)  ###
                    set_by_path(match_dict, path, 1 if decision.match else 0)
                except Exception as e:
                    # If parsing fails, be conservative
                    f.write(f"WARNING!: Exception: {e}")
                    print(f"WARNING!: Exception: {e}")
                    set_by_path(match_dict, path, 0)

            f.write("\n")

        # write space
        f.write("\n\n")

    # Optional Context counters
    compute_context_sums(match_dict, present_fields)
    return match_dict


def eval_for_all_aziende(
    raw_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    output_dir: Path,
    local_evaluator_llm: str,
    present_fields: Optional[Dict[str, List[str]]] = None,
    use_gemini: bool = False,
) -> None:
    """
    Evaluate the Predictions for every Azienda present in pred_data.json using llm as a judge, assigning match score
    and saves the scores in output_dir/match_scores.json and the decisions in output_dir/decision_logs.txt

    Args:
        raw_data: contains raw data (fields values and context_ids/contexts) for the all aziende for which the info is extracted with
                  keys being "name of azienda"
        pred_data: contains pred data (output, context_ids/contexts) for the all aziende for which the info is extracted with
                  keys being "name of azienda"
    """
    LOG_FILE = output_dir / "decision_logs.txt"
    MATCH_SCORE_FILE = output_dir / "match_scores.json"

    with open(
        LOG_FILE, "w", encoding="utf-8"
    ) as f:  # use the write mode to delete previous log for this test
        if use_gemini:
            f.write(f"EVALUATOR_LLM: GEMINI\n")
            print("Evaluatin using GEMINI...")
        else:
            f.write(f"EVALUATOR_LLM: {local_evaluator_llm}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d  %H:%M:%S')}\n")

    match_data_azienda = {}
    for azienda in raw_data.keys():
        print(f"Azienda: {azienda}")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Azienda: {azienda}\n")

        raw_vals = raw_data[azienda]
        pred_v = pred_data[azienda]
        match_data = evaluate_pred_fieldwise(
            reference_obj=raw_vals,
            predicted_obj=pred_v,
            log_file=LOG_FILE,
            local_evaluator_llm=local_evaluator_llm,
            present_fields=present_fields,
            use_gemini=use_gemini,
        )

        match_data_azienda[azienda] = match_data

    with open(MATCH_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(match_data_azienda, f, indent=4, ensure_ascii=False)
