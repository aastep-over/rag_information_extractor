# CONFIG FILE SETTINGS  (Load args form config file)
import argparse
import json
import logging
import time
from pathlib import Path

from rag_info_extractor.utils.load_config import cfgs

from utils.match_pred_rag_QA_llm import eval_for_all_aziende

# Logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Load paths to combined_raw_json and pred_json(output/extracted data json file)"
    )
    parser.add_argument(
        "--combined-raw-json",  # --combined-raw-json "data/jsons/TRAIN/custom_chunks_2/combined_data.json"
        type=str,
        help="Path(relative) to combined_raw_json file",
        required=True,
    )
    parser.add_argument(
        "--pred-json",
        type=str,
        help="Path(relative) to pred_json file which is to be evaluated",
        required=True,
    )
    parser.add_argument(
        "--use-gemini",  # --use-gemini True
        type=bool,
        help="Path(relative) to pred_json file which is to be evaluated",
        default=False,
    )
    parser.add_argument(
        "--save-context",  # --save-context "re_ranked_docs"
        type=str,
        choices=["retrieved_docs", "re_ranked_docs"],
        help="If want to save contexts to decision_logs_qa, which contexts want to save?",
        default="",
    )
    parser.add_argument(
        "--doc-store-large-chunks-path",  # --doc-store-large-chunks-path "data/large_chunks_dbs/TRAIN/custom_chunks_2"
        type=str,
        help="Path (relative) to document store containing larger(parent) chunks.",
    )
    parser.add_argument(
        "--vector-store-path",  #  --vector-store-path "data/vector_dbs/TRAIN/custom_chunks_2"
        type=str,
        help="Path (relative) to vector store (DB) containing smaller embedded (child) chunks.",
    )
    args = parser.parse_args()

    # Read configs
    BASE_DIR = Path(__file__).resolve().parents[2]
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM", "")

    if args.save_context:
        assert (
            args.doc_store_large_chunks_path
        ), "Path to doc_store_large_chunks not passed"
        assert args.vector_store_path, "Path to vector db not passed"
        doc_store_large_chunks_path = str(
            Path(BASE_DIR, args.doc_store_large_chunks_path)
        )
        vector_store_path = str(Path(BASE_DIR, args.vector_store_path))
    else:
        doc_store_large_chunks_path = None
        vector_store_path = None

    # Define all paths
    combined_raw_json = Path(BASE_DIR, args.combined_raw_json)
    pred_json = Path(BASE_DIR, args.pred_json)
    EVAL_OUTPUT_DIR = Path(pred_json).parent

    # Load the raw_data and pred_data jsons
    raw_data = json.loads(combined_raw_json.read_text(encoding="utf-8"))
    pred_data = json.loads(pred_json.read_text(encoding="utf-8"))

    # Match and score pred vs raw and save the results
    logger.info("Evaluating Predicted output w.r.t Raw output using LLM as a judge.")
    eval_for_all_aziende(
        raw_data,
        pred_data,
        EVAL_OUTPUT_DIR,
        EVALUATOR_LLM,
        use_gemini=args.use_gemini,
        context_type=args.save_context,
        doc_store_large_chunks_path=doc_store_large_chunks_path,
        vector_store_path=vector_store_path,
    )

    logger.info(f"Evaluation completed. Results saved to: \t {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    cfgs = cfgs.get("args", {})

    main()

    logger.info(
        f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.perf_counter()-t0))}"
    )
