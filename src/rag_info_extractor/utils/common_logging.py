# scripts/common_logging.py
import datetime
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler


def configure_logging(
    env_var="RAG_LOG_LEVEL", default_level=logging.INFO, logfile=None
):
    # env var e.g. RAG_LOG_LEVEL=DEBUG or INFO
    level_name = os.getenv(env_var, None)
    if level_name:
        level = getattr(logging, level_name.upper(), default_level)
    else:
        level = default_level

    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(
            RotatingFileHandler(
                logfile, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
            )
        )

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    # OPTIONAL: reduce verbosity of noisy third-party libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


# usage in a script:
# from scripts.common_logging import configure_logging
# configure_logging(logfile="rag_dev.log")        # default INFO or env override
# OR for quick debug: RAG_LOG_LEVEL=DEBUG python scripts/ingest_documents.py
