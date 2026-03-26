from langchain_core.documents import Document

# python native
from typing import List, Optional, Dict, Any
import re
from math import isnan
import yaml
from pathlib import Path

from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, IDBasedContextPrecision, IDBasedContextRecall
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall # ! pip install rapidfuzz
from ragas.llms import LangchainLLMWrapper

# from other modules
from rag_info_extractor.utils.llm_connector import OllamaLLM
from rag_info_extractor.utils.load_config import cfgs


# CONFIG FILE SETTINGS  (Load args form config file)
# cfg_path = Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/config.yaml")
# with open(cfg_path, "r", encoding="utf-8") as f:
#     configs = yaml.safe_load(f)

cfgs = cfgs.get("args", {})
EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM", "")

## ------------------------- HELPERS --------------------------------------------------------
#-------------------------------------------------------------------------------------------


evaluator_llm = LangchainLLMWrapper(OllamaLLM(llm_model=EVALUATOR_LLM, temperature=0))

# -------- CONTEXT PRECISION & RECALL FOR 1 QUERY -------------------

# Custom CP function to calculate cp using chunk ids
def context_precision(
    retrieved_contexts: List[int],
    re_ranked_contexts: List[int],
    reference_context_ids: List[int]
) -> Dict[str, float | None]:
    """
    Calculates the rank-aware Context Precision for both retrieved and re-ranked context lists using chunk_ids.
    """
    # no ref context => cp is not defined
    if not reference_context_ids:
        return {"retrieved_CP": None, "re_ranked_CP": None}

    def _get_cp(retrieved_ids: List[int]):
        relevant_count = 0
        precision_sum = 0.0

        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in reference_context_ids:
                relevant_count += 1
                # Precision@i = relevant chunks found so far / total chunks checked so far (idx + 1)
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        # CP = precision_sum / num_of_relevant_chunks
        if relevant_count > 0:
            return precision_sum / relevant_count
        else:
            return 0.0
    
    return {
        "retrieved_CP": _get_cp(retrieved_contexts),
        "re_ranked_CP": _get_cp(re_ranked_contexts)
    }


# Using RAGAS APIs
def context_precision_ragas(
    azienda_name: str,
    question: str,
    answer: str,
    ref_answer: str,
    retrieved_contexts: List[int],
    re_ranked_contexts: List[int],
    reference_context: str,
    reference_context_ids: List[int],
    id_based: bool=True,
    chunks: List[Document]=[],
    use_llm: bool=False,
    verbose: bool = False
) -> Dict[str, float]:
    
    if not id_based and not chunks:
        raise ValueError(
            "Invalid arguments: 'chunks' (List[Document]) must be provided "
            "when 'id_based' is False."
        )

    if verbose:
        print("PRECISION:")
        print(f"Azienda: {azienda_name}")###
        print(f"Question: {question}")###
        print(f"Reference: {reference_context_ids}")###
        print(f"Retrieved: {retrieved_contexts}")###
        print(f"Re-ranked: {re_ranked_contexts}")###
        print()

    if id_based:
        context_precision = IDBasedContextPrecision()
        
        sample_ret = SingleTurnSample(
        retrieved_context_ids=[str(c) for c in retrieved_contexts], 
        reference_context_ids=[str(c) for c in reference_context_ids]
        )

        sample_rerank = SingleTurnSample(
        retrieved_context_ids=[str(c) for c in re_ranked_contexts],
        reference_context_ids=[str(c) for c in reference_context_ids]
        )
        
    else:
        retrieved_contexts_str: List[str] = [' '.join(chunks[i].page_content.split()) for i in retrieved_contexts]
        re_ranked_contexts_str: List[str] = [' '.join(chunks[i].page_content.split()) for i in re_ranked_contexts]
        # Remove spaces/extra chars from strs
        reference_context = ' '.join(reference_context.split())
        reference_context = reference_context.replace("\n", " ")
        retrieved_contexts_str = [c.replace("\n", " ") for c in retrieved_contexts_str]
        re_ranked_contexts_str = [c.replace("\n", " ") for c in re_ranked_contexts_str]


        if use_llm:
            context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
            
            sample_ret = SingleTurnSample(
            user_input=question,
            reference=ref_answer,
            response=answer,
            retrieved_contexts=retrieved_contexts_str
            )

            sample_rerank = SingleTurnSample(
            user_input=question,
            reference=ref_answer,
            response=answer,
            retrieved_contexts=re_ranked_contexts_str
            )
        else:
            context_precision = NonLLMContextPrecisionWithReference()

            sample_ret = SingleTurnSample(
                retrieved_contexts=retrieved_contexts_str, 
                reference_contexts=[reference_context]
            )

            sample_rerank = SingleTurnSample(
                retrieved_contexts=re_ranked_contexts_str, 
                reference_contexts=[reference_context]
            )

    return {
        "retrieved_CP": context_precision.single_turn_score(sample_ret),
        "re_ranked_CP": context_precision.single_turn_score(sample_rerank)
    }

def context_recall(
    azienda_name: str,
    question: str,
    answer: str,
    ref_answer: str,
    retrieved_contexts: List[int],
    re_ranked_contexts: List[int],
    reference_context: str,
    reference_context_ids: List[int],
    id_based: bool=True,
    chunks: List[Document]=[],
    use_llm: bool=False,
    verbose: bool = False
) -> Dict[str, float | None]:
    
    if (not reference_context_ids) and (not reference_context):
        return {"retrieved_CR": None, "re_ranked_CR": None} 

    if not id_based and not chunks:
        raise ValueError(
            "Invalid arguments: 'chunks' (List[Document]) must be provided "
            "when 'id_based' is False."
        )
        
    if verbose:
        print("RECALL:")
        print(f"Azienda: {azienda_name}")###
        print(f"Question: {question}")###
        print(f"Reference: {reference_context_ids}")###
        print(f"Retrieved: {retrieved_contexts}")###
        print(f"Re-ranked: {re_ranked_contexts}")###
        print()
        
    if id_based:
        context_recall = IDBasedContextRecall()

        sample_ret = SingleTurnSample(
        retrieved_context_ids=[str(c) for c in retrieved_contexts], 
        reference_context_ids=[str(c) for c in reference_context_ids]
        )

        sample_rerank = SingleTurnSample(
        retrieved_context_ids=[str(c) for c in re_ranked_contexts],
        reference_context_ids=[str(c) for c in reference_context_ids]
        )
        
    else:
        retrieved_contexts_str: List[str] = [' '.join(chunks[i].page_content.split()) for i in retrieved_contexts]
        re_ranked_contexts_str: List[str] = [' '.join(chunks[i].page_content.split()) for i in re_ranked_contexts]
        # Remove spaces/extra chars from strs
        reference_context = ' '.join(reference_context.split())
        reference_context = reference_context.replace("\n", " ")
        retrieved_contexts_str = [c.replace("\n", " ") for c in retrieved_contexts_str]
        re_ranked_contexts_str = [c.replace("\n", " ") for c in re_ranked_contexts_str]


        if use_llm:
            context_recall = LLMContextRecall(llm=evaluator_llm)

            sample_ret = SingleTurnSample(
            user_input=question,
            reference=ref_answer,
            response=answer,
            retrieved_contexts=retrieved_contexts_str
            )

            sample_rerank = SingleTurnSample(
            user_input=question,
            reference=ref_answer,
            response=answer,
            retrieved_contexts=re_ranked_contexts_str
            )
        else:
            context_recall = NonLLMContextRecall()

            sample_ret = SingleTurnSample(
                retrieved_contexts=retrieved_contexts_str, 
                reference_contexts=[reference_context]
            )

            sample_rerank = SingleTurnSample(
                retrieved_contexts=re_ranked_contexts_str, 
                reference_contexts=[reference_context]
            )
    
    return {
        "retrieved_CR": context_recall.single_turn_score(sample_ret),
        "re_ranked_CR": context_recall.single_turn_score(sample_rerank)
    }



## ------------------------- MAIN FUNCTIONS --------------------------------------------------------
#-------------------------------------------------------------------------------------------

def context_PR_per_company(
    azienda_name: str,
    results_qa: Dict[str, Dict[str, Dict[str, str]]],
    results_contexts: Dict[str, Dict[str, Dict[str, Dict[str, List[int]]]]],
    ref_qa: Dict[str, Dict[str, Dict[str, str]]],
    ref_contexts: Dict[str, Dict[str, str]],
    ref_contexts_ids: Dict[str, Dict[str, Dict[str, List[int]]]],
    use_parent_chunks: bool = False
) -> Dict[str, Any]:
    """
    Calculates Context Precision and Context Recall:
      - averages per subgroup (reported as-is),
      - averages per group (mean over subgroups),
      - averages per company (mean over all subgroups).

    Special handling:
      - If context_recall returns NaN for a subgroup/metric (e.g., no reference context),
        that entry is EXCLUDED from the recall averages. If a group/company has only NaNs
        for a recall metric, its averaged value is set to None.
    """

    per_group = {}

    # Company-level totals and counts
    total_score_cp = {"retrieved_CP": 0.0, "re_ranked_CP": 0.0}
    total_score_cr = {"retrieved_CR": 0.0, "re_ranked_CR": 0.0}
    total_count_cp = {"retrieved_CP": 0,    "re_ranked_CP": 0}     # precision should always count, but keep symmetric
    total_count_cr = {"retrieved_CR": 0,    "re_ranked_CR": 0}

    for group_name, group in results_qa.items():
        per_subgroup = {}

        # Group-level totals and counts
        total_group_score_cp = {"retrieved_CP": 0.0, "re_ranked_CP": 0.0}
        total_group_score_cr = {"retrieved_CR": 0.0, "re_ranked_CR": 0.0}
        count_group_cp       = {"retrieved_CP": 0,    "re_ranked_CP": 0}
        count_group_cr       = {"retrieved_CR": 0,    "re_ranked_CR": 0}

        for sg_name, sg in group.items():
            question, answer = sg['Q'], sg['A']
            
            # Check which chunks to use to calculate precision and recall
            if use_parent_chunks:
                print(f"use parent: {use_parent_chunks}")###
                print(results_contexts[group_name])###
                retrieved_docs = results_contexts[group_name]['retrieved_docs'][sg_name].get("parents", [])
                re_ranked_docs = results_contexts[group_name]['re_ranked_docs'][sg_name].get("parents", [])

                ref_answer = ref_qa[group_name][sg_name]['A']
                reference_context = ref_contexts[group_name][sg_name]
                ref_context_ids = ref_contexts_ids[group_name][sg_name].get("parents", [])

                # if ref_context is missing, set values to None because it shouldn't be counted for calculating mean cp and cr
                if not ref_context_ids:
                    cp = {"retrieved_CP": None, "re_ranked_CP": None}
                    cr = {"retrieved_CR": None, "re_ranked_CR": None}
                else:
                    # cp = context_precision_ragas(
                    #     azienda_name, question, answer, ref_answer,
                    #     retrieved_docs, re_ranked_docs, reference_context, ref_context_ids, 
                    #     use_llm=False
                    # )
                    cp = context_precision(
                        retrieved_docs, re_ranked_docs, ref_context_ids
                    )
                    cr = context_recall(
                        azienda_name, question, answer, ref_answer,
                        retrieved_docs, re_ranked_docs, reference_context, ref_context_ids,
                        use_llm=False
                    )

            else:
                print(f"use parent: {use_parent_chunks}")###
                retrieved_docs = results_contexts[group_name]['retrieved_docs'][sg_name].get("children", [])
                re_ranked_docs = results_contexts[group_name]['re_ranked_docs'][sg_name].get("children", [])

                ref_answer = ref_qa[group_name][sg_name]['A']
                reference_context = ref_contexts[group_name][sg_name]
                ref_context_ids = ref_contexts_ids[group_name][sg_name].get("children", [])

                # if ref_context is missing, set values to None because it shouldn't be counted for calculating mean cp and cr
                if not ref_context_ids:
                    cp = {"retrieved_CP": None, "re_ranked_CP": None}
                    cr = {"retrieved_CR": None, "re_ranked_CR": None}
                else:
                    # cp = context_precision_ragas(
                    #     azienda_name, question, answer, ref_answer,
                    #     retrieved_docs, re_ranked_docs, reference_context, ref_context_ids, 
                    #     use_llm=False
                    # )
                    cp = context_precision(
                        retrieved_docs, re_ranked_docs, ref_context_ids
                    )
                    cr = context_recall(
                        azienda_name, question, answer, ref_answer,
                        retrieved_docs, re_ranked_docs, reference_context, ref_context_ids,
                        use_llm=False
                    )

            # cr looks like:
            # {
            #   "retrieved_CR": context_recall.single_turn_score(sample_ret),
            #   "re_ranked_CR": context_recall.single_turn_score(sample_rerank)
            # }

            per_subgroup[sg_name] = {"Precision": cp, "Recall": cr}

            # ---- accumulate precision (assumed numeric) ----
            for k, v in cp.items():
                if v is not None and not isnan(v):
                    total_group_score_cp[k] += float(v)
                    count_group_cp[k]      += 1
                    total_score_cp[k]      += float(v)
                    total_count_cp[k]      += 1

            # ---- accumulate recall with NaN-skip ----
            for k, v in cr.items():
                if v is not None and not isnan(v):
                    total_group_score_cr[k] += float(v)
                    count_group_cr[k]       += 1
                    total_score_cr[k]       += float(v)
                    total_count_cr[k]       += 1
                # if NaN/None: skip (do not increase counts)

        # ---- averages per group (handle zero counts -> None) ----
        avg_group_score_cp = {
            k: round(total_group_score_cp[k] / count_group_cp[k], 3) if count_group_cp[k] > 0 else None
            for k in total_group_score_cp
        }
        avg_group_score_cr = {
            k: round(total_group_score_cr[k] / count_group_cr[k], 3) if count_group_cr[k] > 0 else None
            for k in total_group_score_cr
        }

        per_group[group_name] = {
            "Precision": avg_group_score_cp,
            "Recall":    avg_group_score_cr,
            "per_subgroup": per_subgroup
        }

    # ---- averages per company (handle zero counts -> None) ----
    avg_score_cp = {
        k: round(total_score_cp[k] / total_count_cp[k], 3) if total_count_cp[k] > 0 else None
        for k in total_score_cp
    }
    avg_score_cr = {
        k: round(total_score_cr[k] / total_count_cr[k], 3) if total_count_cr[k] > 0 else None
        for k in total_score_cr
    }

    return {
        "Precision": avg_score_cp,
        "Recall":    avg_score_cr,
        "per_group": per_group
    }





def _add_to_sums_and_counts(
    sums: Dict[str, float],
    counts: Dict[str, int],
    row: Dict[str, Any],
):
    for k, v in row.items():
        if v is None:
            continue
        # if a float NaN slipped through
        try:
            if isinstance(v, float) and isnan(v):
                continue
        except Exception:
            pass
        sums[k] = sums.get(k, 0.0) + float(v)
        counts[k] = counts.get(k, 0) + 1

def _avg_from_sums_counts(sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, Any]:
    out = {}
    for k, s in sums.items():
        c = counts.get(k, 0)
        out[k] = round(s / c, 3) if c > 0 else None
    return out

def context_PR_overall(
    results_qa: Dict[str, Any],
    results_contexts: Dict[str, Any],
    ref_qa: Dict[str, Any],
    ref_contexts: Dict[str, Any],
    ref_contexts_ids: Dict[str, Any],
    use_parent_chunks: bool = False
) -> Dict[str, Any]:

    # Per-company results
    per_company = {
        name: context_PR_per_company(
            name,
            results_qa[name],
            results_contexts[name],
            ref_qa[name],
            ref_contexts[name],
            ref_contexts_ids[name],
            use_parent_chunks
        )
        for name in ref_qa
    }

    # -------- Overall (company-level) aggregation --------
    cp_sums_overall, cp_counts_overall = {}, {}
    cr_sums_overall, cr_counts_overall = {}, {}

    for res in per_company.values():
        _add_to_sums_and_counts(cp_sums_overall, cp_counts_overall, res["Precision"])
        _add_to_sums_and_counts(cr_sums_overall, cr_counts_overall, res["Recall"])

    avg_overall_score_cp = _avg_from_sums_counts(cp_sums_overall, cp_counts_overall)
    avg_overall_score_cr = _avg_from_sums_counts(cr_sums_overall, cr_counts_overall)

    # -------- Per-group aggregation across companies --------
    group_cp_sums: Dict[str, Dict[str, float]] = {}
    group_cp_counts: Dict[str, Dict[str, int]] = {}
    group_cr_sums: Dict[str, Dict[str, float]] = {}
    group_cr_counts: Dict[str, Dict[str, int]] = {}

    # -------- Per-subgroup aggregation across companies -----
    subgroup_cp_sums: Dict[str, Dict[str, Dict[str, float]]] = {}
    subgroup_cp_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    subgroup_cr_sums: Dict[str, Dict[str, Dict[str, float]]] = {}
    subgroup_cr_counts: Dict[str, Dict[str, Dict[str, int]]] = {}

    for res in per_company.values():
        for g, stats in res["per_group"].items():
            # init dicts
            group_cp_sums.setdefault(g, {})
            group_cp_counts.setdefault(g, {})
            group_cr_sums.setdefault(g, {})
            group_cr_counts.setdefault(g, {})

            # group-level CP/CR
            _add_to_sums_and_counts(group_cp_sums[g], group_cp_counts[g], stats["Precision"])
            _add_to_sums_and_counts(group_cr_sums[g], group_cr_counts[g], stats["Recall"])

            # subgroups
            subgroup_cp_sums.setdefault(g, {})
            subgroup_cp_counts.setdefault(g, {})
            subgroup_cr_sums.setdefault(g, {})
            subgroup_cr_counts.setdefault(g, {})

            for sg_name, sg_stats in stats["per_subgroup"].items():
                subgroup_cp_sums[g].setdefault(sg_name, {})
                subgroup_cp_counts[g].setdefault(sg_name, {})
                subgroup_cr_sums[g].setdefault(sg_name, {})
                subgroup_cr_counts[g].setdefault(sg_name, {})

                _add_to_sums_and_counts(subgroup_cp_sums[g][sg_name], subgroup_cp_counts[g][sg_name], sg_stats["Precision"])
                _add_to_sums_and_counts(subgroup_cr_sums[g][sg_name], subgroup_cr_counts[g][sg_name], sg_stats["Recall"])

    # averages per-group
    group_avg_score_cp = {g: _avg_from_sums_counts(group_cp_sums[g], group_cp_counts[g]) for g in group_cp_sums}
    group_avg_score_cr = {g: _avg_from_sums_counts(group_cr_sums[g], group_cr_counts[g]) for g in group_cr_sums}

    # averages per-subgroup
    subgroup_avg_score_cp = {
        g: {sg: _avg_from_sums_counts(subgroup_cp_sums[g][sg], subgroup_cp_counts[g][sg])
            for sg in subgroup_cp_sums[g]}
        for g in subgroup_cp_sums
    }
    subgroup_avg_score_cr = {
        g: {sg: _avg_from_sums_counts(subgroup_cr_sums[g][sg], subgroup_cr_counts[g][sg])
            for sg in subgroup_cr_sums[g]}
        for g in subgroup_cr_sums
    }

    per_group = {"Precision": group_avg_score_cp, "Recall": group_avg_score_cr}
    per_subgroup = {"Precision": subgroup_avg_score_cp, "Recall": subgroup_avg_score_cr}
    overall = {"Precision": avg_overall_score_cp, "Recall": avg_overall_score_cr}

    return {
        "overall": overall,
        "per_group": per_group,
        "per_company": per_company,
        "per_subgroup": per_subgroup
    }
