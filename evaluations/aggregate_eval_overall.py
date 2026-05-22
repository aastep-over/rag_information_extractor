import sys
from typing import Any, Dict, Tuple, List, Literal
import json
from pathlib import Path
import argparse
import time
from math import isnan


# from other modules
from utils.eval_context_PR import context_PR_overall
from utils.eval_accuracy import accuracy_overall, accuracy_per_company, percentage_of_fields_extracted
from utils.eval_runtime import runtime_overall, runtime_per_company, unformat_time, format_time
from utils.load_aziende_data_dicts import load_company_dicts

def avg_dicts(*dicts):
    """
    Recursively averages the values of arbitrarily many dictionaries 
    that share the exact same nested structure, ignoring None values.
    """
    if not dicts:
        return {}

    result = {}
    first_dict = dicts[0]

    for key in first_dict.keys():
        # If the value is a dictionary, recurse deeper
        if isinstance(first_dict[key], dict):
            result[key] = avg_dicts(*(d[key] for d in dicts))
        else:
            # Extract all non-None values for this key across all dicts
            valid_values = [d[key] for d in dicts if d[key] is not None]
            
            # If we have valid numbers, calculate the average
            if valid_values:
                result[key] = sum(valid_values) / len(valid_values)
            # If all values were None, default to None (or 0 if you prefer)
            else:
                result[key] = None

    return result

def return_single_run_summary(
    companies_match_data: Dict[str, Any],
    companies_match_qa: Dict[str, Any],
    companies_qa: Dict[str, Any],
    companies_contexts: Dict[str, Any],
    companies_ref_qa: Dict[str, Any],
    companies_ref_contexts: Dict[str, Any],
    companies_ref_contexts_ids: Dict[str, Any],
    combined_raw_json: Path,
    combined_pred_json: Path,
    USE_PARENT_CHUNKS: bool = True,
):  

    """Returns summary + detailed results"""

    # Calculate overall accuracy and runtime
    acc_all = accuracy_overall(companies_match_data)
    acc_all_qa = accuracy_overall(companies_match_qa)

    # Calculate Context PR
    if USE_PARENT_CHUNKS:
        context_pr_output_parent = context_PR_overall(
            companies_qa, companies_contexts,
            companies_ref_qa, companies_ref_contexts,
            companies_ref_contexts_ids,
            use_parent_chunks=True
        )

    context_pr_output_children = context_PR_overall(
        companies_qa, companies_contexts,
        companies_ref_qa, companies_ref_contexts,
        companies_ref_contexts_ids
    )

    # Calculate %age of fields extracted
    combined_raw_data = json.loads(combined_raw_json.read_text(encoding="utf-8"))
    combined_pred_data = json.loads(combined_pred_json.read_text(encoding="utf-8"))
    percent_field_extracted = percentage_of_fields_extracted(combined_pred_data, combined_raw_data)

    return (
        acc_all,
        acc_all_qa,
        context_pr_output_parent,
        context_pr_output_children,
        percent_field_extracted,
        companies_match_data,
        companies_match_qa,
    )

def write_summary(
    output_file: Path,
    companies_match_data: Dict[str, Any],
    companies_match_qa: Dict[str, Any],
    acc_all: Dict[str, Any],
    acc_all_qa: Dict[str, Any],
    context_pr_output_parent: Dict[str, Any],
    context_pr_output_children: Dict[str, Any],
    percent_field_extracted: float,
    detailed_result: bool = True,
    USE_PARENT_CHUNKS: bool = True,
    data_to_be_evaluated: Literal["qa", "json", "both"] = "both"
):  

    """Write summary + optional detailed results to the output file."""

    def w(*args, **kwargs):
        """Write a line to the output file instead of printing."""
        text = " ".join(str(a) for a in args)
        f.write(text + ("\n" if kwargs.get("end", "\n") == "\n" else ""))

    with open(output_file, "w", encoding="utf-8") as f:

        # ---------------------------- SUMMARY -----------------------------------------
        w("Use Parent Chunks: TRUE" if USE_PARENT_CHUNKS else "Not Using Parent Chunks")

        w('"""')
        w("SUMMARY:\n")

        # ------------------------ %age of Field extracted -------------------------
        if data_to_be_evaluated in ("both", "json"):
            w("\n\t%age of Field extracted:", f"{percent_field_extracted:.2f}")

            # ------------------------ Accuracy ------------------------
            w("\n\tAccuracy:")
            w("\t\tAvg. Company accuracy:", f"{acc_all['overall']['accuracy']:.3f}")
            w("\t\tAvg. per Group Accuracies:")
            for group, data in acc_all['per_group'].items():
                w(f"\t\t\t{group}: {data['accuracy']:.3f}")

        # ------------------------ Accuracy (rag QA)------------------------
        if data_to_be_evaluated in ("both", "qa"):
            w("\n\tAccuracy (RAG QA):")
            w("\t\tAvg. Company accuracy:", f"{acc_all_qa['overall']['accuracy']:.3f}")
            w("\t\tAvg. per Group Accuracies:")
            for group, data in acc_all_qa['per_group'].items():
                w(f"\t\t\t{group}: {data['accuracy']:.3f}")


        # ------------------------ Context Precision ------------------------
        if USE_PARENT_CHUNKS:
            w("\n\tContext Precision: (PARENT CHUNKS)")
            w("\t\tAvg CP Overall:", context_pr_output_parent['overall'].get("Precision", 0.))
            w("\t\tAvg CP Per Group:")
            for group, data in context_pr_output_parent['per_group']['Precision'].items():
                w(f"\t\t\t{group}: {data}")

        w("\n\tContext Precision: (CHILDREN CHUNKS)")
        w("\t\tAvg CP Overall:", context_pr_output_children['overall'].get("Precision", 0.))
        w("\t\tAvg CP Per Group:")
        for group, data in context_pr_output_children['per_group']['Precision'].items():
            w(f"\t\t\t{group}: {data}")

        # ------------------------ Context Recall ------------------------
        if USE_PARENT_CHUNKS:
            w("\n\tContext Recall: (PARENT CHUNKS)")
            w("\t\tAvg CR Overall:", context_pr_output_parent['overall'].get("Recall", 0.))
            w("\t\tAvg CR Per Group:")
            for group, data in context_pr_output_parent['per_group']['Recall'].items():
                w(f"\t\t\t{group}: {data}")

        w("\n\tContext Recall: (CHILDREN CHUNKS)")
        w("\t\tAvg CR Overall:", context_pr_output_children['overall'].get("Recall", 0.))
        w("\t\tAvg CR Per Group:")
        for group, data in context_pr_output_children['per_group']['Recall'].items():
            w(f"\t\t\t{group}: {data}")

        w('"""')

        # ========================= DETAILS SECTION =========================
        if not detailed_result:
            return

        # ---------------------------- DETAILS -----------------------------------------
        w("\n" + "-" * 120)
        w("-" * 120 + "\n")
        w('"""')
        w("DETAILS:\n")

        # ----------------------------- Accuracy Details -----------------------------
        if data_to_be_evaluated in ("both", "json"):
            w("ACCURACY:\n")
            for c, c_data in companies_match_data.items():
                acc = accuracy_per_company(c_data)
                w("\n\tCompany:", c)
                w("\t\tAvg. Group accuracy:", f"{acc['overall']['accuracy']:.3f}")
                w("\n\t\tPer Group Accuracies:")
                for group, data in acc['per_group'].items():
                    w(f"\t\t\t{group}: {data['accuracy']:.3f}")
                w("\n" + "-" * 40 + " xxxxx " + "-" * 40)

            w("Overall accuracy:")
            w("\tAvg. Company accuracy:", f"{acc_all['overall']['accuracy']:.3f}")
            w("\n\tAvg. per Group Accuracies:")
            for group, data in acc_all['per_group'].items():
                w(f"\t\t{group}: {data['accuracy']:.3f}")
        
        # ----------------------------- Accuracy Details (rag QA) -----------------------------
        if data_to_be_evaluated in ("both", "qa"):
            w("\n\nACCURACY (RAG QA):\n")
            for c, c_data in companies_match_qa.items():
                acc = accuracy_per_company(c_data)
                w("\n\tCompany:", c)
                w("\t\tAvg. Group accuracy:", f"{acc['overall']['accuracy']:.3f}")
                w("\n\t\tPer Group Accuracies:")
                for group, data in acc['per_group'].items():
                    w(f"\t\t\t{group}: {data['accuracy']:.3f}")
                w("\n" + "-" * 40 + " xxxxx " + "-" * 40)

            w("Overall accuracy (RAG QA):")
            w("\tAvg. Company accuracy:", f"{acc_all_qa['overall']['accuracy']:.3f}")
            w("\n\tAvg. per Group Accuracies:")
            for group, data in acc_all_qa['per_group'].items():
                w(f"\t\t{group}: {data['accuracy']:.3f}")

            w('"""')

        # ----------------------------- Context Precision Details -----------------------------
        w("\n" + "-" * 120)
        w("-" * 120 + "\n")
        w('"""')

        if USE_PARENT_CHUNKS:
            w("CONTEXT PRECISION: (PARENT CHUNKS)\n")
            for c, c_data in context_pr_output_parent['per_company'].items():
                w("\n\tCompany:", c)
                w("\t\tAvg. Group CP:", c_data['Precision'])
                w("\n\t\tPer Group CP:")
                for group, data in c_data['per_group'].items():
                    w(f"\t\t\t{group}: {data['Precision']}")
                w("\n" + "-" * 40 + " xxxxx " + "-" * 40)

            w("Overall Context Precision:\n")
            w("\tAvg CP Overall:", context_pr_output_parent['overall'].get("Precision", 0.))
            w("\tAvg CP Per Group:")
            for group, data in context_pr_output_parent['per_group']['Precision'].items():
                w(f"\t\t{group}: {data}")

            w("\tAvg CP Per Sub-Group:")
            for group, data in context_pr_output_parent['per_subgroup']['Precision'].items():
                w(f"\t\t{group}:")
                for sg, sg_data in data.items():
                    w(f"\t\t\t{sg}: {sg_data}")

        # children chunks
        w("\nCONTEXT PRECISION: (CHILDREN CHUNKS)\n")
        for c, c_data in context_pr_output_children['per_company'].items():
            w("\n\tCompany:", c)
            w("\t\tAvg. Group CP:", c_data['Precision'])
            w("\n\t\tPer Group CP:")
            for group, data in c_data['per_group'].items():
                w(f"\t\t\t{group}: {data['Precision']}")
            w("\n" + "-" * 40 + " xxxxx " + "-" * 40)

        w("Overall Context Precision:\n")
        w("\tAvg CP Overall:", context_pr_output_children['overall'].get("Precision", 0.))
        w("\tAvg CP Per Group:")
        for group, data in context_pr_output_children['per_group']['Precision'].items():
            w(f"\t\t{group}: {data}")

        w("\tAvg CP Per Sub-Group:")
        for group, data in context_pr_output_children['per_subgroup']['Precision'].items():
            w(f"\t\t{group}:")
            for sg, sg_data in data.items():
                w(f"\t\t\t{sg}: {sg_data}")

        w('"""')

        # ----------------------------- Context Recall Details -----------------------------
        w("\n" + "-" * 120)
        w("-" * 120 + "\n")
        w('"""')

        if USE_PARENT_CHUNKS:
            w("CONTEXT Recall:\n")
            for c, c_data in context_pr_output_parent['per_company'].items():
                w("\n\tCompany:", c)
                w("\t\tAvg. Group CR:", c_data['Recall'])
                w("\n\t\tPer Group CR:")
                for group, data in c_data['per_group'].items():
                    w(f"\t\t\t{group}: {data['Recall']}")
                w("\n" + "-" * 40 + " xxxxx " + "-" * 40)

            w("Overall Context Recall:\n")
            w("\tAvg CR Overall:", context_pr_output_parent['overall'].get("Recall", 0.))
            w("\tAvg CR Per Group:")
            for group, data in context_pr_output_parent['per_group']['Recall'].items():
                w(f"\t\t{group}: {data}")

            w("\tAvg CR Per Sub-Group:")
            for group, data in context_pr_output_parent['per_subgroup']['Recall'].items():
                w(f"\t\t{group}:")
                for sg, sg_data in data.items():
                    w(f"\t\t\t{sg}: {sg_data}")

        # children chunks
        w("\nCONTEXT Recall:\n")
        for c, c_data in context_pr_output_children['per_company'].items():
            w("\n\tCompany:", c)
            w("\t\tAvg. Group CR:", c_data['Recall'])
            w("\n\t\tPer Group CR:")
            for group, data in c_data['per_group'].items():
                w(f"\t\t\t{group}: {data['Recall']}")
            w("\n" + "-" * 40 + " xxxxx " + "-" * 40)

        w("Overall Context Recall:\n")
        w("\tAvg CR Overall:", context_pr_output_children['overall'].get("Recall", 0.))
        w("\tAvg CR Per Group:")
        for group, data in context_pr_output_children['per_group']['Recall'].items():
            w(f"\t\t{group}: {data}")

        w("\tAvg CR Per Sub-Group:")
        for group, data in context_pr_output_children['per_subgroup']['Recall'].items():
            w(f"\t\t{group}:")
            for sg, sg_data in data.items():
                w(f"\t\t\t{sg}: {sg_data}")

        w('"""')

    





if __name__ == "__main__":

    from rag_info_extractor.utils.load_config import cfgs
    import os

    parser = argparse.ArgumentParser(
        description="Load paths to combined_raw_json and combined_pred_json(output/extracted data json file)"
    )
    parser.add_argument(
        "--combined-raw-json", # "data/jsons/TRAIN/fixed_size_chunks/combined_data.json"
        type=str,
        help="Path(relative) to combined_raw_json file",
        required=True
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        help="Path(relative) to directory containing all the runs to be aggregated (e.g. runs/TRAIN/ablation_overall_llms/gemma3-4b)",
        required=True
    )
    parser.add_argument(
        "--data-to-be-evaluated",
        type=str,
        choices=["qa", "json", "both"],
        help="Which data(s) to evaluate:  only rag_qa or only extracted_json or rag_qa + extracted_json",
        default="both"
    ) # TODO: THINK OF CLEANER WAY TO IMPLEMENT IT

    args = parser.parse_args()
    
    t0 = time.time()

    # Load configs
    cfgs = cfgs.get("args", {})
    BASE_DIR = Path(__file__).resolve().parents[1]
    run_names = [f for f in os.listdir(Path(BASE_DIR, args.runs_dir)) if os.path.isdir(Path(BASE_DIR, args.runs_dir, f))]
    print(args.runs_dir)

    # Obtain paths for raw_json, combined_pred_json and match_scores_json
    combined_raw_json = Path(BASE_DIR, args.combined_raw_json)
    combined_pred_jsons = [Path(BASE_DIR, args.runs_dir, run, "pred.json") for run in run_names]
    match_scores_jsons = [Path(BASE_DIR, args.runs_dir, run, "match_scores.json") for run in run_names]
    match_scores_qa_jsons = [Path(BASE_DIR, args.runs_dir, run, "match_scores_qa.json") for run in run_names]

    assert combined_raw_json.exists(), f"File not found: {combined_raw_json}"
    for i in range(len(combined_pred_jsons)):
        assert combined_pred_jsons[i].exists(), f"File not found: {combined_pred_jsons[i]}"
        assert match_scores_jsons[i].exists(), f"File not found: {match_scores_jsons[i]}"
        assert match_scores_qa_jsons[i].exists(), f"File not found: {match_scores_qa_jsons[i]}"

    # Calculate for each run
    acc_all_all_runs = []
    acc_all_qa_all_runs = []
    context_pr_output_parent_all_runs = []
    context_pr_output_children_all_runs = []
    companies_match_data_all_runs = []
    companies_match_qa_all_runs = []
    percent_field_extracted_all_runs = []

    for i in range(len(combined_pred_jsons)):
        (
            companies_match_data,
            companies_match_qa,
            companies_pred_qa,
            companies_raw_qa,
            companies_raw_contexts,
            companies_pred_contexts,
            companies_raw_contexts_ids,
            companies_pred_contexts_ids,
            companies_runtimes,
        ) = load_company_dicts(combined_raw_json, combined_pred_jsons[i], match_scores_jsons[i], match_scores_qa_jsons[i])


        (
            acc_all,
            acc_all_qa,
            context_pr_output_parent,
            context_pr_output_children,
            percent_field_extracted,
            companies_match_data,
            companies_match_qa,
        ) = return_single_run_summary(
            companies_match_data = companies_match_data,
            companies_match_qa = companies_match_qa,
            companies_qa = companies_pred_qa,
            companies_contexts = companies_pred_contexts_ids,
            companies_ref_qa = companies_raw_qa,
            companies_ref_contexts = companies_raw_contexts, 
            companies_ref_contexts_ids = companies_raw_contexts_ids,
            combined_raw_json = combined_raw_json,
            combined_pred_json = combined_pred_jsons[i],
            USE_PARENT_CHUNKS = True,
        )

        # Append to all_runs
        acc_all_all_runs.append(acc_all)
        acc_all_qa_all_runs.append(acc_all_qa)
        context_pr_output_parent_all_runs.append(context_pr_output_parent)
        context_pr_output_children_all_runs.append(context_pr_output_children)
        percent_field_extracted_all_runs.append(percent_field_extracted)
        companies_match_data_all_runs.append(companies_match_data)
        companies_match_qa_all_runs.append(companies_match_qa)

    # Average all runs
    acc_all = avg_dicts(*acc_all_all_runs)
    acc_all_qa = avg_dicts(*acc_all_qa_all_runs)
    # runtime_all = avg_dicts(*runtime_all_all_runs) # TODO: fix avg_dict error
    context_pr_output_parent = avg_dicts(*context_pr_output_parent_all_runs)
    context_pr_output_children = avg_dicts(*context_pr_output_children_all_runs)
    companies_match_data = avg_dicts(*companies_match_data_all_runs)
    companies_match_qa = avg_dicts(*companies_match_qa_all_runs)
    # companies_runtime = avg_dicts(*companies_runtime_all_runs) # TODO: fix avg_dict error
    percent_field_extracted = sum(percent_field_extracted_all_runs) / len(percent_field_extracted_all_runs)

  

    # Save summary to aggregate_summary
    write_summary(
        output_file = BASE_DIR / args.runs_dir / "aggregate_summary.txt",
        companies_match_data = companies_match_data,
        companies_match_qa=companies_match_qa,
        acc_all = acc_all,
        acc_all_qa = acc_all_qa,
        context_pr_output_parent = context_pr_output_parent,
        context_pr_output_children = context_pr_output_children,
        percent_field_extracted = percent_field_extracted,
        detailed_result = True,
        USE_PARENT_CHUNKS = True,
        data_to_be_evaluated = args.data_to_be_evaluated
    )
