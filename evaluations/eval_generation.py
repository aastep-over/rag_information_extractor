from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from typing import Dict, Any, List, Tuple, Literal, Optional
import json, yaml
import copy
from tqdm import tqdm
import re
import textwrap
import argparse
from pathlib import Path
import time

from rag_info_extractor.utils.llm_connector import OllamaLLM
from utils.load_aziende_data_dicts import load_company_dicts
from utils.eval_accuracy import accuracy_overall

# logging relative
import logging
from rag_info_extractor.utils.common_logging import configure_logging
logger = logging.getLogger(__name__)



# ---------------------------------- GENERATE ANSWERS FROM RAW CONTEXT ----------------------------------
# ----------------------------------                                   ----------------------------------

def generate_answer_from_raw_context(
    companies_ref_qa: Dict[str, Dict[str, Dict[str, Dict[str, str]]]],
    companies_ref_contexts_ids: Dict[str, Dict[str, Dict[str, Dict[str, List[int]]]]],
    chunks: List[Document],
    llm_model: str,
    use_parent_chunks: bool = True,
    camps_already_filled: Dict[str, Dict[str, Dict[str, Literal["yes", "no"]]]] = {}
):  
    """
    Generate answer using llm_model provided the correct context.

    Args:
        companies_ref_qa: dict of questions and raw(ground-truth) answers
        companies_ref_contexts_ids: 
    """
    llm = OllamaLLM(
        llm_model = llm_model
    )

    # Prompt
    system_prompt = textwrap.dedent("""\
        Sei un analista specializzato in statuti societari.
        ISTRUZIONI:
        - Usa esclusivamente il CONTESTO per rispondere.
        - Mantieni la risposta il più concisa possibile.
        - Se l’informazione non è presente nel contesto; scrivi esattamente: "Non ho trovato la risposta nei documenti forniti".
        - Non inventare né inferire oltre ciò che è esplicitamente scritto.;
            Niente elenco puntato, niente preamboli.

        CONTESTO:
        {context}

        DOMANDA:
        {question}
    """)
    

    overall_generated_qa = {}
    for company_name, company_data in tqdm(companies_ref_qa.items(), desc="Generating Answers for:"):
        print("\n\nCompany: ", company_name)###
        per_company_qa = {}
        for group_name, group_data in company_data.items():
            print("\n\tGroup: ", group_name)###
            per_group_qa = {}
            for subgroup_name, subgroup_data in group_data.items():

                # skip if answer already generated
                if camps_already_filled and (camps_already_filled[company_name][group_name][subgroup_name] == "yes"):
                    continue

                question = subgroup_data.get("Q", "")
                if use_parent_chunks:
                    context_ids = companies_ref_contexts_ids[company_name][group_name][subgroup_name].get("parents", [])
                else:
                    context_ids = companies_ref_contexts_ids[company_name][group_name][subgroup_name].get("children", [])
                
                # generate ans
                context = "\n\n".join([chunks[i].page_content for i in context_ids])

                # Update prompt with context and question
                prompt_content = system_prompt.replace("{context}", context)
                prompt_content = prompt_content.replace("{question}", question)

                
                answer: AIMessage = llm.invoke(
                    output_format = "text",
                    memory = prompt_content,
                    num_predict = 500,
                    temperature = 0,
                    use_cache = False
                ) # type: ignore
                
                per_group_qa[subgroup_name] = {"Q": question, "A": answer}

                # save check point: subgroup
                per_company_qa[group_name] = per_group_qa
                overall_generated_qa[company_name] = per_company_qa
                print("\t\textracted info for SUB-GROUP: ", subgroup_name)###
                with open("./last_run_eval_generation.json", 'w', encoding="utf-8") as f:
                    json.dump(overall_generated_qa, f, indent=4, ensure_ascii=False)

            # save check point: group
            per_company_qa[group_name] = per_group_qa
            overall_generated_qa[company_name] = per_company_qa
            print("\textracted info for GROUP: ", group_name)###
            with open("./last_run_eval_generation.json", 'w', encoding="utf-8") as f:
                json.dump(overall_generated_qa, f, indent=4, ensure_ascii=False)
        
        # save check point: company
        overall_generated_qa[company_name] = per_company_qa
        print("extracted info for COMPANY: ", company_name)###
        with open("./last_run_eval_generation.json", 'w', encoding="utf-8") as f:
            json.dump(overall_generated_qa, f, indent=4, ensure_ascii=False)
    
    # save results
    with open("./last_run_eval_generation.json", 'w', encoding="utf-8") as f:
        json.dump(overall_generated_qa, f, indent=4, ensure_ascii=False)

    return overall_generated_qa
        
# ---------------------------- XXX ----------------------------




# ---------------------------------- EVALUATE GENERATED ANSWERS ----------------------------------
# ----------------------------------                                   ----------------------------------

# ---------------------------- HELPERS ----------------------------

# ---- MATCH TEMPLATE ----
MATCH_TEMPLATE = {
    'CompensoAmministratori': {
        'Rimborso': 0,
        'IndennitaAnnuale': 0,
        'IndennitaCessazione': 0
    },
    'BilanciUtili': {
        'PercentualeRiservaLegale': 0,
        'CapitaleSociale': 0,
        'TermineApprovazioneBilancio': 0,
        'DataChiusuraEsercizio': 0,
        'UtiliResidui': 0,
    },
    'InfoGenerali': {
        "Durata": 0,
    }
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

system_prompt = textwrap.dedent("""\
    SYSTEM:
                                
    Sei un valutatore severo ma equo per un singolo campo.
    Stabilisci se PREDICTED corrisponde semanticamente a REFERENCE:
    - Accetta differenze banali di formattazione (1,000 vs 1000; 31/12/2024 vs 2024-12-31).
    - yes/true/sì → True; no/false → False.
    - Per numeri/percentuali/durate, confronta il valore sottostante.
    - Se uno dei due valori è mancante mentre l’altro è presente, è una mancata corrispondenza.
    {format_instructions}
    
    HUMAN:
                                
    Field: {field_path}
    REFERENCE: {reference}
    PREDICTED: {predicted}
    Return your decision.
""")

parser = PydanticOutputParser(pydantic_object=FieldDecision)
PROMPT_CONTENT: str = system_prompt.replace(
    "{format_instructions}",
    f"\nFORMAT:\n{parser.get_format_instructions()}"
)



# =======================================================
# 5) Main function: field-by-field with one LLM call each
# =======================================================
def evaluate_pred_fieldwise(
    reference_obj: Dict[str, Any],
    predicted_obj: Dict[str, Any],
    llm_model: str,
    present_fields: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Returns a filled match_data dict with 1/0 per leaf.
    One LLM call per field (skips call if fast_equal == True).
    """
    match_dict = copy.deepcopy(MATCH_TEMPLATE)
    
    llm = OllamaLLM(llm_model)

    for path in leaf_paths(match_dict):
        ref_v = get_by_path(reference_obj, path).get("A")
        pred_v = get_by_path(predicted_obj, path).get("A")

        print(f"Node: {path}")###
        print(f"ref_v: {ref_v}")###
        print(f"pred_v: {pred_v}")###

        # Fast path: exact/normalized equality -> no LLM call
        if fast_equal(ref_v, pred_v):
            print("Fast Path...")###
            set_by_path(match_dict, path, 1)
            continue
        
        # LLM decision (single field)
        payload = {
            "field_path": " / ".join(path),
            "reference": json.dumps(ref_v, ensure_ascii=False) if isinstance(ref_v, (dict, list)) else ref_v,
            "predicted": json.dumps(pred_v, ensure_ascii=False) if isinstance(pred_v, (dict, list)) else pred_v,
        }
        for k, v in payload.items():
            PROMPT_CONTENT = PROMPT_CONTENT.replace("{" + f"{k}" + "}", f"{v}") # type: ignore  # Prompt_CONTENT is always str.

        try:
            decision: FieldDecision = llm.invoke(
                output_format = "structured",
                memory = PROMPT_CONTENT,
                use_cache = False,
                num_predict = 64,
                temperature = 0
            ) # type: ignore

            print(decision)###
            set_by_path(match_dict, path, 1 if decision.match else 0)
        except Exception as e:
            # If parsing fails, be conservative
            print(f"WARNING!: Exception: {e}")
            set_by_path(match_dict, path, 0)

    return match_dict

# MAIN evaluate generated answer
def evaluate_generation_from_raw_context(
    companies_ref_qa: Dict[str, Dict[str, Dict[str, Dict[str, str]]]],
    companies_generated_qa: Dict[str, Dict[str, Dict[str, Dict[str, str]]]],
    llm_model: str
) -> Dict[str, Dict[str, Dict[str, int]]]:
    
    overall_scores_qa = {}
    for company_name in companies_ref_qa:
        print("\n", "-"*40, f"COMPANY NAME: {company_name}", "-"*40, "\n")
        per_company_scores = evaluate_pred_fieldwise(
            reference_obj = companies_ref_qa.get(company_name, {}),
            predicted_obj = companies_generated_qa.get(company_name, {}),
            llm_model = llm_model
        )

        overall_scores_qa[company_name] = per_company_scores

    return overall_scores_qa


# ---------------------------- XXX ----------------------------




# ---------------------------------- CALCULATE ACCURACY ----------------------------------
# ----------------------------------                                   ----------------------------------

# Re-define _group_leaf_counts since this dictionary has one less layer
def _group_leaf_counts(group: Dict[str, Any]) -> Tuple[int, int]:
    """Return (correct, total) over all leaf fields in a group, excluding 'Context' key."""
    correct = 0
    total = 0
    for sub_name, v in group.items():
        if isinstance(v, int):  # 0/1
            total += 1
            correct += int(v)
    return correct, total


def write_summary(
    scores: Dict[str, Any],
    output_file: str,
    llm_model: str
):
    acc_all = accuracy_overall(scores)

    def w(*args, **kwargs):
        """Write a line to the output file instead of printing."""
        text = " ".join(str(a) for a in args)
        f.write(text + ("\n" if kwargs.get("end", "\n") == "\n" else ""))

    with open(output_file, "w", encoding="utf-8") as f:
        w(f"\nLLM: {llm_model}")
        w("\n\tAvg. Company accuracy: ", f"{acc_all['overall']['accuracy']:.3f}")
        w(f"\tAvg. per Group Accuracies:")
        for group, data in acc_all['per_group'].items():
                w(f"\t\t{group}: {data['accuracy']:.3f}")
        w("\n", "-"*50, "xxxxx", "-"*50, "\n")



if __name__ == "__main__":

    from rag_info_extractor.utils.load_config import cfgs
    from rag_info_extractor.utils.embedder import HFEmbedder
    

    # Parse args
    parser = argparse.ArgumentParser(
        description="Load dataset JSON files from ./data/<dataset_type> and build company dictionaries."
    )
    parser.add_argument(
        "dataset_type",
        choices=["TRAIN", "VAL", "TEST"],
        help="Which dataset folder to load from ./data/<dataset_type> (TRAIN | VAL | TEST).",
    )
    parser.add_argument(
        "llm_model",
        type=str,
        choices=["gemma3:4b", "mistral:7b-instruct", "llama3.2:3b-instruct-q8_0", "llama3.2:3b-instruct-q4_0"],
        help="LLM model_name from config",
    )
    parser.add_argument(
        "use_parent_chunks",
        type=bool,
        choices=[True, False],
        default=True,
        help="Use parent/large chunks as context or child/small chunks"
    )

    args = parser.parse_args()

    # Load configs
    t0 = time.time()

    # # CONFIG FILE SETTINGS:
    # cfg_path = Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/config.yaml")
    # with open(cfg_path, "r", encoding="utf-8") as f:
    #     configs = yaml.safe_load(f)

    cfgs = cfgs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = Path(__file__).resolve().parents[1]
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM", "mistral:7b-instruct")
    
    
    # from arg parser 
    dataset_root = Path(BASE_DIR, "tests", "data")
    dataset_dir = dataset_root / args.dataset_type

    outputs_file = Path(BASE_DIR, "tests", "output_to_test.json")

    LLM_MODEL = str(Path(args.llm_model))
    USE_PARENT_CHUNKS = args.use_parent_chunks


    # Use parent chunks/ children chunks
    if USE_PARENT_CHUNKS:
        DOC_STORE_LARGE_CHUNKS_PATH = Path(BASE_DIR) / "data" / "large_chunks_dbs" / args.dataset_type / args.chunk_type
    
        # Load the Parent chunks
        doc_store_page_content = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content") 
        doc_store_metadata = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/metadata")
        num_files = sum(1 for p in DOC_STORE_LARGE_CHUNKS_PATH.iterdir() if p.is_file())
        keys = range(0, num_files)

        parent_page_contents: List[bytes | None] = doc_store_page_content.mget(list(map(str, keys))) 
        parent_metadatas: List[bytes | None] = doc_store_metadata.mget(list(map(str, keys)))

        chunks: List[Document] = [ # type: ignore
                            Document(
                                page_content=bytes.decode(p, encoding="utf-8"),
                                metadata=json.loads(bytes.decode(m, encoding="utf-8"))
                            )
                            for p, m in zip(parent_page_contents, parent_metadatas) if (p and m)
                        ]
    
    else:
        VECTOR_STORE_PATH = Path(BASE_DIR) / "data" / "vector_dbs" / args.dataset_type / args.chunk_type
        # embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
        #                           encode_kwargs={"normalize_embeddings": True})
        embedding = HFEmbedder(normalize_embeddings=True)
        vector_store = Chroma(embedding_function=embedding,
                            persist_directory=VECTOR_STORE_PATH,
                            collection_name="pdf_chunks")
        child_page_contents: List[str] = vector_store.get()["documents"]
        child_metadatas: List[Dict[str, Any]] = vector_store.get()["metadatas"]
        chunks: List[Document] = sorted(
            [Document(page_content=p, metadata=m) for p, m in zip(child_page_contents, child_metadatas)],
            key=lambda x: x.metadata.get("chunk_id", 0)
        )


    # Load data dicts
    (
        companies_match_data,
        companies_match_qa,
        companies_pred_qa,
        companies_raw_qa, #
        companies_raw_contexts,
        companies_pred_contexts,
        companies_raw_contexts_ids,#
        companies_pred_contexts_ids,
        companies_runtimes,
    ) = load_company_dicts(dataset_dir, outputs_file)

    # Generate answers from raw context
    companies_generated_qa = generate_answer_from_raw_context(
    companies_ref_qa = companies_raw_qa,
    companies_ref_contexts_ids = companies_raw_contexts_ids,
    chunks = chunks,
    llm_model = LLM_MODEL,
    use_parent_chunks = True # depend on DOC_STORE_LARGE_CHUNKS_PATH
    )
    
    # Evaluate the answers 
    generation_scores = evaluate_generation_from_raw_context(
        companies_ref_qa = companies_raw_qa,
        companies_generated_qa = companies_generated_qa,
        llm_model = EVALUATOR_LLM
    )

    # save the scores
    write_summary(
        scores = generation_scores,
        output_file = "./results/eval_generation.json",
        llm_model = LLM_MODEL
    ) 

