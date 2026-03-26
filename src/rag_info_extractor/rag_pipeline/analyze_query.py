
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from rapidfuzz import fuzz

# Python native
from typing import List
import re
import textwrap

# Import from modules
from rag_info_extractor.rag_pipeline.utils import CompanyMatcher, match_azienda_name
from rag_info_extractor.utils.llm_connector import OllamaLLM

# Logging
import logging
logger = logging.getLogger(__name__)


class Search(BaseModel):
    """Search query."""
    query: str = Field(description="Solo la query di ricerca ottimizzata; nessun testo extra; non includere il nome della società.")
    azienda: str = Field(description="nome completo ufficiale della società di quale sono richieste le informazioni.")


def analyze_query(
    question: str,
    llm: OllamaLLM,
    nome_azienda: str="",
    azienda_name_records: List[str]=[]
) -> Search:
    """
    Re-phrase query to improve it
    
    nome_azienda (str): name of the società for which the information needs to be extracted 
    """
    logger.info("\n --------------- NODE: __analyze_query__ ------------------------\n")###
    system_prompt = textwrap.dedent("""\
        SYSTEM:
        Sei un ottimizzatore di query per un sistema RAG su statuti societari.

        Genera UNA sola query testuale per motori di ricerca/BM25/hybrid, NON per database.
        Rispondi SOLO con le chiavi i) 'query': contenente la query di ricerca ottimizzata in Italiano e ii) 'azienda': nome completo ufficiale della società di quale sono richieste le informazioni.
                                    
        HUMAN:
    """)

    prompt_content = system_prompt + question

    response: Search = llm.invoke(
        output_format = "structured",
        info_schema = Search,
        memory = prompt_content,
        num_predict = 128,
        temperature = 0,
        cache = False
    ) # type: ignore


    # Ensure nome azienda predicted by llm matches in db (closest) 
    response.azienda = match_azienda_name(response.azienda, azienda_name_records)
    
    # Find nome_azienda from query if not returned by model
    if not response.azienda:
        matcher = CompanyMatcher(azienda_name_records)
        res = matcher.match(question, min_score=78, scorer=fuzz.token_set_ratio, top_k=1)
        if res:
            response.azienda = res[0]['canonical']
        else:
            response.azienda = ""    

    # format name and query    
    response.azienda = response.azienda.lower()
    # remove name of azienda from the query to avoid including name in semantic matching
    pattern = re.compile(r"(?i)(?<!\w)(?:{}|name|company_name)(?!\w)".format(re.escape(response.azienda))) 
    response.query = pattern.sub("", response.query).strip()

    logger.debug("Nome Azienda in 'analyze_query': ", response.azienda)###
    logger.debug("Optimized Query: ", response.query)###

    return response




if __name__ == "__main__":
    import time
    from rag_info_extractor.utils.common_logging import configure_logging
    import argparse

    t0 = time.time()

    question = "Agli amministratori spetta il rimborso delle spese? Quali spese sono incluse? Cerca per la società: 2KIN D SRL"
    llm = OllamaLLM(
        llm_model="gemma3:4b"
    )
    azienda_name_records = ['2kind srl', 'cnc world s.r.l.', 'inverso srl', 'compagnie de participation hotelliere et touristique s.r.l.', 'harmin s.r.l.']

    opt_query = analyze_query(
        question = question,
        llm = llm,
        nome_azienda="",
        azienda_name_records = azienda_name_records
    )

    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f"Optimized Query: {opt_query}")

    logger.info(f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}")