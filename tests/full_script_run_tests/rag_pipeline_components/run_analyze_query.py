import argparse
import asyncio
import logging
import time

from rag_info_extractor.rag_pipeline_components.analyze_query import (
    aanalyze_query,
    analyze_query,
)
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.llm_connector import OllamaLLM
from rag_info_extractor.utils.load_config import cfgs

logger = logging.getLogger(__name__)


def main():
    # 1. Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)

    # 2. CONFIG FILE SETTINGS:
    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")

    # 3. Setup query, azienda_names
    question = "Agli amministratori spetta il rimborso delle spese? Quali spese sono incluse? Cerca per la società: 2KIN D SRL"
    llm = OllamaLLM(llm_model=EMBEDDING_MODEL_NAME)
    azienda_name_records = [
        "2kind srl",
        "cnc world s.r.l.",
        "inverso srl",
        "compagnie de participation hotelliere et touristique s.r.l.",
        "harmin s.r.l.",
    ]

    if RUN_ASYNC:
        logger.info("Async...")
        opt_query = asyncio.run(
            aanalyze_query(
                question=question,
                llm=llm,
                nome_azienda="",
                azienda_name_records=azienda_name_records,
                use_google_api=True,
            )
        )
    else:
        logger.info("Sync...")
        opt_query = analyze_query(
            question=question,
            llm=llm,
            nome_azienda="",
            azienda_name_records=azienda_name_records,
            use_google_api=True,
        )

    logger.info(f"Optimized Query: {opt_query}")


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})
    RUN_ASYNC = True

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )
