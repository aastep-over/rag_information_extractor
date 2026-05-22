import argparse
import datetime
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.embedder import HFEmbedder
from rag_info_extractor.utils.load_config import cfgs

from utils.find_raw_chunks_ids import (
    analyze_raw_contexts_coverage,
    build_combined_chunk_json,
    find_azienda_chunk_ids,
    format_context_ids_lists_in_json,
    load_children_chunks_from_chroma,
    load_parent_chunks_from_dir,
)

logger = logging.getLogger(__name__)
load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-type",
        type=str,
        choices=[
            "custom_chunks",
            "fixed_size_chunks",
            "semantic_chunks",
            "custom_chunks_2",
        ],
        help="Chunking method used to extract context from",
        required=True,
    )
    parser.add_argument("--dataset", type=str, choices=["TEST", "TRAIN"], required=True)
    parser.add_argument(
        "--function",
        type=str,
        choices=[
            "find_azienda_chunk_ids",
            "build_combined_chunk_json",
            "analyze_raw_contexts_coverage",
        ],
        help="""
        Function to run in the script:
        find_azienda_chunk_ids: Find raw chunk ids for a single azienda
        build_combined_chunk_json: Find raw chunk ids for all aziende and update combined_raw_data.json file in data/jsons/
        analyze_raw_contexts_coverage: Verify if raw chunk ids correctly found
        """,
        required=True,
    )
    parser.add_argument("--azienda", default="medicare salute & servizi s.r.l.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )  # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Logging for {"-"*30} evaluations/utils/find_raw_chunks_ids.py")

    # Obtain raw_data dict
    combined_raw_json_path = Path(
        BASE_DIR, f"data/jsons/{args.dataset}/{args.chunk_type}/combined_data.json"
    )

    # Obtain larger chunks (parent chunks) as Document
    DOC_STORE_LARGE_CHUNKS_PATH = (
        f"{BASE_DIR}/data/large_chunks_dbs/{args.dataset}/{args.chunk_type}"
    )
    parent_chunks = load_parent_chunks_from_dir(f"{DOC_STORE_LARGE_CHUNKS_PATH}")
    logger.info(
        f"Loaded parent {len(parent_chunks)} chunks from: {DOC_STORE_LARGE_CHUNKS_PATH}"
    )

    # Obtain child/small chunks from vector db (chroma)
    embedding = HFEmbedder(normalize_embeddings=True)
    VECTOR_STORE_PATH = f"{BASE_DIR}/data/vector_dbs/{args.dataset}/{args.chunk_type}"
    children_chunks = load_children_chunks_from_chroma(VECTOR_STORE_PATH, embedding)
    logger.info(
        f"Loaded {len(children_chunks)} children chunks from: {VECTOR_STORE_PATH}"
    )

    if args.function == "find_azienda_chunk_ids":
        # 1. Find raw chunk ids for a single azienda
        azienda = args.azienda
        logger.warning(
            f"Finding raw chunks ids for a single azienda...'{azienda}'.\nFor Different Azienda, pass --azienda argument."
        )
        combined_raw_data = json.loads(
            combined_raw_json_path.read_text(encoding="utf-8")
        )
        data_azienda = combined_raw_data.get(azienda, {})
        raw_contxts_ids = find_azienda_chunk_ids(
            raw_contexts=data_azienda,
            azienda_name=azienda,
            parent_chunks=parent_chunks,
            children_chunks=children_chunks,
        )
        raw_contxts_ids = format_context_ids_lists_in_json(raw_contxts_ids)
        print(raw_contxts_ids)

    elif args.function == "build_combined_chunk_json":
        # 2. Find raw chunk ids for all aziende and update combined_raw_data.json file in data/jsons/
        logger.info("Finding raw chunks ids...")
        build_combined_chunk_json(
            combined_raw_json_path=combined_raw_json_path,
            parent_chunks=parent_chunks,
            children_chunks=children_chunks,
        )
        logger.info(f"Found raw chunks and updated {combined_raw_json_path}")

    else:
        # 3. Verify if raw chunk ids correctly found
        results_below_threshold = analyze_raw_contexts_coverage(
            combined_raw_json_path=str(combined_raw_json_path),
            parent_chunks_dir=DOC_STORE_LARGE_CHUNKS_PATH,
            children_chunks=children_chunks,
            threshold=0.7,
        )

    with open("output_temp.json", "w", encoding="utf-8") as f:
        f.write("Output for find_raw_chunks_ids.py\n")
        f.write(
            f"Date: {datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Function ran: analyze_raw_contexts_coverage\n\n\n")

        json.dump(results_below_threshold, f, indent=4, ensure_ascii=False)
    print(json.dumps(results_below_threshold, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    t0 = time.time()

    # CONFIG FILE SETTINGS:
    cfgs = cfgs.get("args", {})
    BASE_DIR = Path(__file__).resolve().parents[2]

    main()

    logger.info(
        f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}"
    )

# Workflow for Verifying raw_chunks:
# 1. Read output_temp.json saved byfind_raw_chunks_id.py to check which fileds have been outputed with low coverage threshold score
# 2. Read the chunk of the idxs for which cov_threshold output is low in DEMO.ipynb (for parent and/or child)
# 3. Compare and update combined_data.json if needed


## Verified for TRAIN/custom_chunks
