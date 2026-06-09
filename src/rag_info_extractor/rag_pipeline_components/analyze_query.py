# Python native
import json
import logging
import os
import re
import textwrap
from typing import List

from dotenv import load_dotenv

load_dotenv()


# GEMINI API 
from google import genai
from google.genai import types

# Third-party libraries
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from tenacity import retry, wait_random_exponential

from rag_info_extractor.info_schema.utils import return_default_json

# Import from modules
from rag_info_extractor.rag_pipeline_components.utils import (
    CompanyMatcher,
    match_azienda_name,
)
from rag_info_extractor.utils.llm_connector import OllamaLLM

# Logging
logger = logging.getLogger(__name__)


# =======================================
# Extract with GOOGLE API
# =======================================
@retry(wait=wait_random_exponential(min=1, max=60))
def extract_with_GOOGLE_API(prompt_content: str, info_schema: BaseModel):

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    retry_options = types.HttpRetryOptions(
        initial_delay=2.0,
        max_delay=60.0,
        exp_base=2.0,
        attempts=10,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )

    client = genai.Client(http_options=types.HttpOptions(retry_options=retry_options))

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = client.models.generate_content(
        model=os.environ.get("ANALYZE_QUERY__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.1,
            # response_json_schema=info_schema.model_json_schema()#['properties']
        ),
    )

    if response.text:
        response_as_json_code_block = re.match(
            r"^```json\s*([\s\S]*?)\s*```$", response.text, flags=re.MULTILINE
        )
        if response_as_json_code_block:
            clean_string = response_as_json_code_block.group(1)
            # clean_string = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        else:
            clean_string = response.text.strip()
        response_json = json.loads(clean_string)
    else:
        logger.exception("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(
            info_schema.model_json_schema()["properties"]
        )

    return response_json


# Async version
@retry(wait=wait_random_exponential(min=1, max=60))
async def aextract_with_GOOGLE_API(
    prompt_content: str,
    info_schema: BaseModel,
):
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    retry_options = types.HttpRetryOptions(
        initial_delay=2.0,
        max_delay=60.0,
        exp_base=2.0,
        attempts=10,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )

    client = genai.Client(http_options=types.HttpOptions(retry_options=retry_options))

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = await client.aio.models.generate_content(
        model=os.environ.get("ANALYZE_QUERY__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.1,
            # response_json_schema=info_schema.model_json_schema()['properties']
        ),
    )

    if response.text:
        response_as_json_code_block = re.match(
            r"^```json\s*([\s\S]*?)\s*```$", response.text, flags=re.MULTILINE
        )
        if response_as_json_code_block:
            clean_string = response_as_json_code_block.group(1)
            # clean_string = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        else:
            clean_string = response.text.strip()
        response_json = json.loads(clean_string)
    else:
        logger.exception("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(
            info_schema.model_json_schema()["properties"]
        )

    return response_json


ANALYZE_QUERY_SYSTEM_PROMPT = textwrap.dedent(
    """\
    SYSTEM:
        Sei un ottimizzatore di query per un sistema RAG su statuti societari.

    OBIETTIVO:
        Dato un input utente, genera un JSON valido seguendo lo schema fornito.

    REGOLE:
        1. OUTPUT
        - Restituisci SOLO un JSON valido
        - Nessun testo extra, nessuna spiegazione

        2. USO DELLO SCHEMA
        - Usa la DESCRIZIONE SCHEMA per capire cosa inserire in ogni campo
        - Rispetta tutte le regole presenti nelle description dei campi
        - Non aggiungere chiavi non richieste

        3. ESTRAZIONE
        - Se il nome della società è presente, estrailo correttamente
        - La query NON deve contenere il nome della società

        4. ROBUSTEZZA
        - Non inventare informazioni
        - Se un campo non è presente, usa il default previsto

    DESCRIZIONE SCHEMA:
        {module_json_schema_properties}

    ESEMPI:
        Input: "Qual è il quorum assembleare di Alfa S.p.A.?"
        Output:
        {"query": "quorum assembleare statuto", "azienda": "Alfa S.p.A."}

        Input: "Come funziona il diritto di recesso?"
        Output:
        {"query": "diritto di recesso statuto società", "azienda": ""}

    HUMAN:
"""
)
# l'ultima riga: Rispondi SOLO con le chiavi i) 'query': contenente la query di ricerca ottimizzata in Italiano che non dovrebbe contenere il nome della società e ii) 'azienda': nome completo ufficiale della società di quale sono richieste le informazioni.


class Search(BaseModel):
    """Search query."""

    query: str = Field(
        default="",
        description="""
            - Deve essere ottimizzata per ricerca BM25/hybrid
            - Usa parole chiave rilevanti e specifiche
            - Rimuovi stopwords e frasi inutili
            - NON includere il nome della società
            - Mantieni il significato della domanda
            - Mantieni lingua originale (italiano)
            - Non aggiungere informazioni non presenti nella domanda
        """,
    )
    azienda: str = Field(
        default="",
        description="""
            - Se l’utente menziona una società → estrai la denominazione completa
            - Se NON è presente → usa stringa vuota ""
            - Non inventare nomi
        """,
    )


def analyze_query(
    question: str,
    llm: OllamaLLM,
    nome_azienda: str = "",
    azienda_name_records: List[str] = [],
    use_google_api: bool = False,
) -> Search:
    """
    Re-phrase query to improve it

    nome_azienda (str): name of the società for which the information needs to be extracted
    """
    logger.info(
        "\n --------------- NODE: __analyze_query__ ------------------------\n"
    )  ###

    system_prompt = ANALYZE_QUERY_SYSTEM_PROMPT.replace(
        "{module_json_schema_properties}",
        json.dumps(
            Search.model_json_schema()["properties"], indent=2, ensure_ascii=False
        ),
    )

    prompt_content = system_prompt + question

    if use_google_api:
        response = extract_with_GOOGLE_API(
            prompt_content=prompt_content, info_schema=Search  # type: ignore
        )

        logger.info(f"RESPONSE FROM GOOGLE API: {response}")
        response = Search.model_validate(response)  # type: ignore
    else:
        response: Search = llm.invoke(
            output_format="structured",
            info_schema=Search,
            memory=prompt_content,
            num_predict=128,
            temperature=0,
            cache=False,
        )  # type: ignore

    # Ensure nome azienda predicted by llm matches in db (closest)
    response.azienda = match_azienda_name(response.azienda, azienda_name_records)

    # Find nome_azienda from query if not returned by model
    if not response.azienda:
        matcher = CompanyMatcher(azienda_name_records)
        res = matcher.match(
            question, min_score=78, scorer=fuzz.token_set_ratio, top_k=1
        )
        if res:
            response.azienda = res[0]["canonical"]
        else:
            response.azienda = ""

    # format name and query
    response.azienda = response.azienda.lower()
    # remove name of azienda from the query to avoid including name in semantic matching
    pattern = re.compile(
        r"(?i)(?<!\w)(?:{}|name|company_name)(?!\w)".format(re.escape(response.azienda))
    )
    response.query = pattern.sub("", response.query).strip()

    logger.debug("Nome Azienda in 'analyze_query': ", response.azienda)  ###
    logger.debug("Optimized Query: ", response.query)  ###

    return response


async def aanalyze_query(
    question: str,
    llm: OllamaLLM,
    nome_azienda: str = "",
    azienda_name_records: List[str] = [],
    use_google_api: bool = False,
) -> Search:
    """
    Re-phrase query to improve it

    nome_azienda (str): name of the società for which the information needs to be extracted
    """
    logger.info(
        "\n --------------- NODE: (async) __analyze_query__ ------------------------\n"
    )  ###

    system_prompt = ANALYZE_QUERY_SYSTEM_PROMPT.replace(
        "{module_json_schema_properties}",
        json.dumps(
            Search.model_json_schema()["properties"], indent=2, ensure_ascii=False
        ),
    )
    prompt_content = system_prompt + question

    if use_google_api:
        response = await aextract_with_GOOGLE_API(
            prompt_content=prompt_content, info_schema=Search  # type: ignore
        )
        logger.info(f"RESPONSE FROM GOOGLE API: {response}")
        response = Search.model_validate(response)  # type: ignore
    else:
        response: Search = await llm.ainvoke(
            output_format="structured",
            info_schema=Search,
            memory=prompt_content,
            num_predict=128,
            temperature=0,
            cache=False,
        )  # type: ignore

    # Ensure nome azienda predicted by llm matches in db (closest) # No need for to_thread since very fast for small azienda_name_records
    response.azienda = match_azienda_name(response.azienda, azienda_name_records)

    # Find nome_azienda from query if not returned by model
    if not response.azienda:
        matcher = CompanyMatcher(azienda_name_records)
        # No need for to_thread since very fast for small azienda_name_records for matcher.match
        res = matcher.match(
            question, min_score=78, scorer=fuzz.token_set_ratio, top_k=1
        )
        if res:
            response.azienda = res[0]["canonical"]
        else:
            response.azienda = ""

    # format name and query
    response.azienda = response.azienda.lower()
    # remove name of azienda from the query to avoid including name in semantic matching
    pattern = re.compile(
        r"(?i)(?<!\w)(?:{}|name|company_name)(?!\w)".format(re.escape(response.azienda))
    )
    response.query = pattern.sub("", response.query).strip()

    logger.debug("Nome Azienda in 'analyze_query': ", response.azienda)  ###
    logger.debug("Optimized Query: ", response.query)  ###

    return response

