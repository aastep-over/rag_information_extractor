import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# GEMINI API 
from google import genai
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# from other modules
from rag_info_extractor.utils.embedder import HFEmbedder
from tenacity import retry, wait_random_exponential

logger = logging.getLogger(__name__)

# ---- MATCH TEMPLATE ----
MATCH_TEMPLATE = {
    "COMPENSO_DEGLI_AMMINISTRATORI": {
        "Rimborso": 0,
        "IndennitaAnnuale": 0,
        "IndennitaCessazione": 0,
    },
    "BILANCI_E_UTILI": {
        "PercentualeRiservaLegale": 0,
        "CapitaleSociale": 0,
        "TermineApprovazioneBilancio": 0,
        "DataChiusuraEsercizio": 0,
        "UtiliResidui": 0,
        # 'Context': 0,
    },
    "INFO_GENERALI": {
        "Durata": 0,
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
    Converts the following dicts to --> MATCH TEMPLATE: (removes the key values and output)

    combined_raw_json:
        {
            'COMPENSO_DEGLI_AMMINISTRATORI': {
                "raw_qa": {
                    'Rimborso': {
                        'Q': '',
                        'A': '',
                    },
                    ...
            },
            'BILANCI_E_UTILI': {},
            ...
        }

    combined_pred_json:
        {
            'COMPENSO_DEGLI_AMMINISTRATORI': {
                "rag_qa": {
                    'Rimborso': {
                        'Q': '',
                        'A': '',
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
            group_data = group.get("raw_qa", {})
        elif data_type == "pred":
            group_data = group.get("rag_qa", {})

        match_template[group_name] = group_data

    return match_template


def convert_context_pred_json_to_match_template(
    context_pred_json: Dict[str, Any],
    context_type: Literal["retrieved_docs", "re_ranked_docs"],
):
    """
    Converts the following dicts to --> MATCH TEMPLATE: (removes the key values and output)

    context_pred_json:
        {
            'COMPENSO_DEGLI_AMMINISTRATORI': {
                "context_type": {
                        'Rimborso': {
                            'parents': [...],
                            'children': [...],
                    },
                    ...
            },
            'BILANCI_E_UTILI': {},
            ...
        }
    """
    match_template = {}
    for group_name, group in context_pred_json.items():
        match_template[group_name] = {}
        if context_type == "retrieved_docs":
            group_data = group.get("retrieved_docs", {"parents": [], "children": []})
        elif context_type == "re_ranked_docs":
            group_data = group.get("re_ranked_docs", {"parents": [], "children": []})

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
        description="True se la risposta PREDICTED risponde correttamente alla QUESTION basandosi sui fatti della REFERENCE; altrimenti False.",
    )


EVAL_SYSTEM = (
    "Sei un valutatore esperto, severo ma imparziale.\n"
    "Il tuo compito è stabilire se la risposta generata (PREDICTED) risponde correttamente alla "
    "domanda (QUESTION) basandosi ESCLUSIVAMENTE sulle informazioni della verità di base (REFERENCE).\n\n"
    "Regole di valutazione:\n"
    "- Restituisci True se PREDICTED contiene le informazioni chiave presenti in REFERENCE.\n"
    "- Restituisci True se PREDICTED è una stringa vuota mentre REFERENCE dice 'Non ho trovato la risposta nei documenti forniti.'.\n"
    "- Accetta risposte discorsive: se PREDICTED è più lunga o usa parole diverse (sinonimi, parafrasi) ma il significato è lo stesso di REFERENCE, è True.\n"
    "- Ignora le differenze di formattazione (es. 1.000 vs 1000, 31/12/2024 vs 2024-12-31, maiuscole/minuscole).\n"
    "- Restituisci False se PREDICTED contraddice REFERENCE, se omette il nocciolo della risposta corretta, o se inventa informazioni che alterano il significato (allucinazioni).\n"
    "- Restituisci False se PREDICTED dice 'non lo so' mentre REFERENCE contiene la risposta.\n"
    "{format_instructions}"
)


EVAL_USER = (
    "Field: {field_path}\n"
    "QUESTION: {question}\n"
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
    question: str,
    reference: str,
    predicted: str,
    system_prompt: str,
    user_prompt: str,
):

    # replace field, Question, Reference and Predicted in EVAL_USER prompt
    name_azienda = re.findall(r".*Nome della società: (.*)", question)
    if name_azienda:
        question_senza_nome = question.replace(
            f"Nome della società: {name_azienda[0]}", ""
        )
        reference_senza_nome = reference.replace(name_azienda[0], "<nome_azienda>")
        predicted_senza_nome = predicted.replace(name_azienda[0], "<nome_azienda>")
    else:
        question_senza_nome = question
        reference_senza_nome = reference
        predicted_senza_nome = predicted

    user_prompt = user_prompt.replace("{field_path}", " / ".join(path))
    user_prompt = user_prompt.replace("{question}", question_senza_nome)
    user_prompt = user_prompt.replace("{reference}", reference_senza_nome)
    user_prompt = user_prompt.replace("{predicted}", predicted_senza_nome)

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
    context_type: Optional[Literal["retrieved_docs", "re_ranked_docs"]] = None,
    doc_store_large_chunks_path: Optional[str] = None,
    vectordb_page_contents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns a filled match_data dict with 1/0 per leaf.
    One LLM call per field (skips call if fast_equal == True).

    Args:
        reference_obj: contains raw data (fields values and context_ids/contexts) for the one azienda for which the info is extracted with
        predicted_obj: contains predicted data (fields values and context_ids/contexts) for the one azienda for which the info is extracted with
        log_file: path to the file where the logs will be saved
        present_fields: optional list of fields to be evaluated
        use_gemini: if True, the evaluation will be done with the GEMINI API
        context_type: optional type of context to be evaluated
        doc_store_large_chunks_path: path to the directory where the large chunks are stored
        vectordb_page_contents: list of page contents from the vector database sorted by chunk_id
    """
    match_dict = copy.deepcopy(MATCH_TEMPLATE)

    if context_type:
        assert (
            doc_store_large_chunks_path and vectordb_page_contents
        ), "doc_store_large_chunks_path and vectordb_page_contents are required when context_type is not None"
        context_pred_obj = convert_context_pred_json_to_match_template(
            predicted_obj, context_type
        )  # Has to come before converting predicted_obj

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
            if context_type:
                context_v = get_by_path(context_pred_obj, path)
                context_passed = ""
                if context_v.get("parents"):
                    for parent_id in context_v.get("parents"):
                        with open(
                            f"{doc_store_large_chunks_path}/page_content/{parent_id}",
                            encoding="utf-8",
                        ) as large_doc_file:
                            context_passed += large_doc_file.read()
                            context_passed += "\n\n"
                if context_v.get("children"):
                    for child_id in context_v.get("children"):
                        context_passed += f"{vectordb_page_contents[child_id]}\n"  # type: ignore
                        context_passed += "\n\n"

            f.write(f"\nNode: {path}\n")
            f.write(f"question: {ref_v['Q']}\n")
            f.write(f"ref_v: {ref_v['A']}\n")
            f.write(f"pred_v: {pred_v['A']}\n")
            f.write(f"context_passed_to_LLM:\n {context_passed}\n\n")

            print(f"Node: {path}")  ###
            print(f"question: {ref_v['Q']}")
            print(f"ref_v: {ref_v['A']}")  ###
            print(f"pred_v: {pred_v['A']}")  ###
            print(f"context_passed_to_LLM:\n {context_passed[:10]}")  ###

            # Fast path: exact/normalized equality -> no LLM call
            if fast_equal(ref_v["A"], pred_v["A"]) or (
                not ref_v["A"]
                and pred_v["A"] == "Non ho trovato la risposta nei documenti forniti."
            ):
                f.write("Fast Path...\n")
                print("Fast Path...")  ###
                set_by_path(match_dict, path, 1)
                continue

            # If pred_v empty or ref_v empty while other is not then set it to 0 -> no LLM call (since otherwise prev. if-statement will be executed)
            if not ref_v["A"]:
                f.write("NOT MATCHED...\n")
                print("NOT MATCHED...")
                set_by_path(match_dict, path, 0)
                continue
            if not pred_v["A"]:
                f.write("NOT MATCHED...\n")
                print("NOT MATCHED...")
                set_by_path(match_dict, path, 0)
                continue

            if use_gemini:
                try:
                    decision_gemini = evaluate_with_GEMINI(
                        path=path,
                        question=ref_v["Q"],
                        reference=ref_v["A"],
                        predicted=pred_v["A"],
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
                    "question": ref_v["Q"],
                    "reference": ref_v["A"],
                    "predicted": pred_v["A"],
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
    context_type: Optional[Literal["retrieved_docs", "re_ranked_docs"]] = None,
    doc_store_large_chunks_path: Optional[str] = None,
    vector_store_path: Optional[str] = None,
) -> None:
    """
    Evaluate the Predictions for every Azienda present in pred_data.json using llm as a judge, assigning match score
    and saves the scores in output_dir/match_scores.json and the decisions in output_dir/decision_logs.txt

    Args:
        raw_data: contains raw data (fields values and context_ids/contexts) for the all aziende for which the info is extracted with
                  keys being "name of azienda"
        pred_data: contains pred data (output, context_ids/contexts) for the all aziende for which the info is extracted with
                  keys being "name of azienda"
        present_fields: optional list of fields to be evaluated
        use_gemini: if True, the evaluation will be done with the GEMINI API
        context_type: optional type of context to be evaluated
        doc_store_large_chunks_path: path to the directory where the large chunks are stored
        vector_store_path: path to the vector database to retrieve the context contents
    """

    if context_type:
        assert (
            doc_store_large_chunks_path and vector_store_path
        ), "doc_store_large_chunks_path and vector_store_path are required when context_type is not None"
        embedding = HFEmbedder(normalize_embeddings=True)
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=vector_store_path,
            collection_name="pdf_chunks",
        )
        page_contents_list, metadatas_list = (
            vector_store.get()["documents"],
            vector_store.get()["metadatas"],
        )
        metadatas_chunk_ids_tuples = [
            (i, m.get("chunk_id")) for i, m in enumerate(metadatas_list)
        ]
        metadatas_chunk_ids_tuples.sort(key=lambda x: x[1])

        # sort as per chunk_ids
        idxs_for_sorted_list = [x[0] for x in metadatas_chunk_ids_tuples]
        page_contents = [page_contents_list[idx] for idx in idxs_for_sorted_list]
    else:
        page_contents = None

    LOG_FILE = output_dir / "decision_logs_qa.txt"
    MATCH_SCORE_FILE = output_dir / "match_scores_qa.json"

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
            context_type=context_type,
            doc_store_large_chunks_path=doc_store_large_chunks_path,
            vectordb_page_contents=page_contents,
        )

        match_data_azienda[azienda] = match_data

    with open(MATCH_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(match_data_azienda, f, indent=4, ensure_ascii=False)
