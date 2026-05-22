import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from rag_info_extractor.rag_pipeline_components.generator import agenerate, generate
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
    logger.info(
        f"Logging for {"-"*30} rag_information_extractor/src/rag_info_extractor/rag_pipeline/retrieve.py"
    )

    # 2. CONFIG FILE SETTINGS:
    LLM_MODEL = cfgs.get("LLM_MODEL")
    BASE_DIR = Path(__file__).resolve().parents[3]

    # 3. Load env_vars
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # 4. Setup Query and Aziende (EXAMPLE)
    QUESTION = "Agli amministratori spetta il rimborso delle spese?"
    CONTEXTS = [
        Document(
            id="0d290c5c-68c0-4330-8bf3-36640fe144bd",
            metadata={
                "creationDate": "D:20230711100553+02'00'",
                "modDate": "D:20230711100553+02'00'",
                "total_pages": 7,
                "end": 14981,
                "producer": "OAPDFPrinter        ",
                "keywords": "",
                "child_id": 17,
                "azienda": "2kind srl",
                "creator": "",
                "format": "PDF 1.5",
                "pattern_name": "art_keyword",
                "title": "Agli Amministratori spetta, oltre al rimborso delle spese",
                "subject": "",
                "filename": "8048909650002.pdf",
                "start": 14830,
                "parent_id": 19,
                "header": "Art.19.– Agli Amministratori spetta, oltre al rimborso delle spese",
                "trapped": "",
                "source": "8048909650002.pdf",
                "author": "",
                "chunk_id": 17,
            },
            page_content="Art.19.– Agli Amministratori spetta, oltre al rimborso delle spese\nsostenute in ragione del loro ufficio, un compenso eventuale determinato dai\nsoci.",
        ),
        Document(
            id="f021a117-cc81-49cf-916b-a0ea57b911c2",
            metadata={
                "creator": "",
                "azienda": "2kind srl",
                "total_pages": 7,
                "source": "8048909650002.pdf",
                "filename": "8048909650002.pdf",
                "start": 13925,
                "modDate": "D:20230711100553+02'00'",
                "format": "PDF 1.5",
                "keywords": "",
                "child_id": 13,
                "chunk_id": 13,
                "trapped": "",
                "header": "Art.15. -  Per la validità delle deliberazioni del consiglio è necessaria la",
                "author": "",
                "title": "-  Per la validità delle deliberazioni del consiglio è necessaria la",
                "pattern_name": "art_keyword",
                "creationDate": "D:20230711100553+02'00'",
                "subject": "",
                "end": 14084,
                "producer": "OAPDFPrinter        ",
                "parent_id": 15,
            },
            page_content="Art.15. -  Per la validità delle deliberazioni del consiglio è necessaria la\npresenza ed il voto favorevole della maggioranza degli amministratori in\ncarica.",
        ),
        Document(
            id="c13c41ca-2a7f-45f2-9a34-e25410462fdd",
            metadata={
                "trapped": "",
                "modDate": "D:20230711100553+02'00'",
                "end": 14830,
                "start": 14549,
                "pattern_name": "art_keyword",
                "creationDate": "D:20230711100553+02'00'",
                "title": "La firma e la rappresentanza sociale spettano all'Amministratore",
                "chunk_id": 18,
                "format": "PDF 1.5",
                "parent_id": 18,
                "creator": "",
                "child_id": 18,
                "subject": "",
                "header": "Art.18.- La firma e la rappresentanza sociale spettano all'Amministratore",
                "total_pages": 7,
                "filename": "8048909650002.pdf",
                "azienda": "2kind srl",
                "producer": "OAPDFPrinter        ",
                "author": "",
                "keywords": "",
                "source": "8048909650002.pdf",
            },
            page_content="Art.18.- La firma e la rappresentanza sociale spettano all'Amministratore\nUnico, ai coamministratori disgiuntamente o congiuntamente, o, nel caso in\ncui esista il Consiglio di Amministrazione, al Presidente di quest’ultimo e\nagli amministratori delegati, nei limiti della delega.",
        ),
        Document(
            id="2f1775e3-2cd0-45f1-b7a1-feabc15e1bf5",
            metadata={
                "keywords": "",
                "end": 13925,
                "total_pages": 7,
                "producer": "OAPDFPrinter        ",
                "header": "Art.14. – Il Consiglio di Amministrazione viene convocato dal presidente",
                "pattern_name": "art_keyword",
                "parent_id": 14,
                "child_id": 11,
                "chunk_id": 11,
                "creationDate": "D:20230711100553+02'00'",
                "title": "– Il Consiglio di Amministrazione viene convocato dal presidente",
                "modDate": "D:20230711100553+02'00'",
                "subject": "",
                "author": "",
                "start": 13677,
                "format": "PDF 1.5",
                "trapped": "",
                "filename": "8048909650002.pdf",
                "source": "8048909650002.pdf",
                "creator": "",
                "azienda": "2kind srl",
            },
            page_content="Art.14. – Il Consiglio di Amministrazione viene convocato dal presidente\ncon comunicazione scritta trasmessa, almeno tre giorni prima dell'adunanza,\na ciascun amministratore e nei casi di urgenza con telegramma da spedirsi\nalmeno un giorno prima.",
        ),
        Document(
            id="2905c0f5-b9f9-47ee-87b0-965c539dd400",
            metadata={
                "producer": "OAPDFPrinter        ",
                "format": "PDF 1.5",
                "header": "Art.13.– Il Consiglio di Amministrazione sceglie fra i suoi membri un",
                "azienda": "2kind srl",
                "pattern_name": "art_keyword",
                "end": 13677,
                "creator": "",
                "chunk_id": 10,
                "author": "",
                "trapped": "",
                "total_pages": 7,
                "source": "8048909650002.pdf",
                "filename": "8048909650002.pdf",
                "keywords": "",
                "child_id": 10,
                "title": "Il Consiglio di Amministrazione sceglie fra i suoi membri un",
                "parent_id": 13,
                "subject": "",
                "modDate": "D:20230711100553+02'00'",
                "start": 13307,
                "creationDate": "D:20230711100553+02'00'",
            },
            page_content="Art.13.– Il Consiglio di Amministrazione sceglie fra i suoi membri un\nPresidente, se questi non è nominato dai soci, ed eventualmente, un\nVice-Presidente che sostituisce il Presidente in caso di assenza o\nimpedimento di quest'ultimo.\n Per tutte le ipotesi di cessazione, rinuncia e sostituzione degli\namministratori si applicano gli articoli 2385 e 2386 codice civile.",
        ),
        Document(
            id="a63fb470-684a-4bf5-817e-9c0765dec258",
            metadata={
                "start": 14981,
                "producer": "OAPDFPrinter        ",
                "pattern_name": "art_keyword",
                "modDate": "D:20230711100553+02'00'",
                "filename": "8048909650002.pdf",
                "creator": "",
                "author": "",
                "child_id": 19,
                "header": "Art.20.- Gli amministratori non possono assumere la qualità di soci",
                "subject": "",
                "end": 15399,
                "title": "Gli amministratori non possono assumere la qualità di soci",
                "creationDate": "D:20230711100553+02'00'",
                "chunk_id": 19,
                "format": "PDF 1.5",
                "azienda": "2kind srl",
                "trapped": "",
                "total_pages": 7,
                "source": "8048909650002.pdf",
                "keywords": "",
                "parent_id": 20,
            },
            page_content="Art.20.- Gli amministratori non possono assumere la qualità di soci\nillimitatamente responsabili in società concorrenti, né esercitare un'attività\nconcorrente per conto proprio o di terzi, né essere amministratori o direttori\ngenerali in una società concorrente, salvo autorizzazione dell'assemblea.\nL'amministratore che non osservi il presente divieto può essere revocato  e\nrisponde dei danni.\n ORGANO DI CONTROLLO",
        ),
        Document(
            id="bd370da1-3006-4096-9f3c-4efada77d1dd",
            metadata={
                "azienda": "2kind srl",
                "parent_id": 17,
                "source": "8048909650002.pdf",
                "trapped": "",
                "title": "– All'organo amministrativo spettano i poteri più ampi per",
                "producer": "OAPDFPrinter        ",
                "child_id": 15,
                "format": "PDF 1.5",
                "creationDate": "D:20230711100553+02'00'",
                "total_pages": 7,
                "chunk_id": 15,
                "header": "Art.17. – All'organo amministrativo spettano i poteri più ampi per",
                "end": 14549,
                "modDate": "D:20230711100553+02'00'",
                "pattern_name": "art_keyword",
                "creator": "",
                "filename": "8048909650002.pdf",
                "subject": "",
                "keywords": "",
                "start": 14315,
                "author": "",
            },
            page_content="Art.17. – All'organo amministrativo spettano i poteri più ampi per\nl'amministrazione della società tanto in via ordinaria che in via straordinaria,\ntranne ciò che per legge o dal presente statuto è demandato alle decisioni dei\nsoci.",
        ),
        Document(
            id="3a6bf210-21a7-4cb4-b6c5-05737ddd00fc",
            metadata={
                "format": "PDF 1.5",
                "author": "",
                "creationDate": "D:20230711100553+02'00'",
                "header": "Art.16. - Il consiglio di amministrazione può delegare le proprie",
                "source": "8048909650002.pdf",
                "producer": "OAPDFPrinter        ",
                "subject": "",
                "title": "- Il consiglio di amministrazione può delegare le proprie",
                "trapped": "",
                "azienda": "2kind srl",
                "modDate": "D:20230711100553+02'00'",
                "creator": "",
                "keywords": "",
                "parent_id": 16,
                "total_pages": 7,
                "child_id": 14,
                "pattern_name": "art_keyword",
                "end": 14315,
                "chunk_id": 14,
                "start": 14084,
                "filename": "8048909650002.pdf",
            },
            page_content="Art.16. - Il consiglio di amministrazione può delegare le proprie\nattribuzioni ad un o più dei suoi membri determinandone all'atto della\nnomina i poteri con le limitazioni di cui all'articolo 2381 c.c.\nPOTERI DEGLI AMMINISTRATORI",
        ),
    ]

    # 5. Run Generate
    llm = OllamaLLM(llm_model=LLM_MODEL, temperature=0)
    if USE_GOOGLE_API:
        logger.info("Using Google API")
        if RUN_ASYNC:
            logger.info("Async...")
            answer = asyncio.run(
                agenerate(
                    question=QUESTION, contexts=CONTEXTS, llm=llm, use_google_api=True
                )
            )
        else:
            logger.info("Sync...")
            answer = generate(
                question=QUESTION, contexts=CONTEXTS, llm=llm, use_google_api=True
            )
    else:
        logger.info("Using Local Ollama")
        if RUN_ASYNC:
            logger.info("Async...")
            answer = asyncio.run(
                agenerate(
                    question=QUESTION, contexts=CONTEXTS, llm=llm, use_google_api=False
                )
            )
        else:
            logger.info("Sync...")
            answer = generate(
                question=QUESTION, contexts=CONTEXTS, llm=llm, use_google_api=False
            )

    # 6. Save the output in output_temp.txt
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("## OUTPUT FOR: generator.py\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d  %H:%M:%S')}\n")
        f.write(f"Question: {QUESTION} \n")
        f.write(f"Answer: {answer}\n")

        f.write(f"\n{"x"*100}\n")
        f.write("CONTEXTS: \n\n")

        for i, c in enumerate(CONTEXTS):
            f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
            f.write(f"CHUNK ID: {c.metadata.get("chunk_id")}\n")
            f.write(f"{c.page_content}\n\n")


if __name__ == "__main__":
    t0 = time.time()
    cfgs = cfgs.get("args", {})
    RUN_ASYNC = True
    USE_GOOGLE_API = True

    main()

    logger.info(
        "Total time taken to run the script: %s",
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)),
    )
