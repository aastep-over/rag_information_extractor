# import pytest

import asyncio
import json
import logging
import os
import re
import textwrap

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential

load_dotenv()

from rag_info_extractor.info_schema.schemas.bilanci_e_utili import (
    CapitaleSociale,
    TermineApprovazioneBilancio,
)
from rag_info_extractor.info_schema.utils import return_default_json
from rag_info_extractor.utils.llm_connector import OllamaLLM

logging.basicConfig(level=logging.INFO)

EXTRACTOR_SYSTEM_PROMPT = textwrap.dedent(
    """\
    SYSTEM:

    Sei un normalizzatore di RISPOSTE per un sistema RAG su statuti societari.

    DESCRIZIONE MODELLO:
    {sub_module_description}

    ISTRUZIONI:
    - Compila i campi del modello esclusivamente usando la RISPOSTA fornita (non usare il contesto).
    - Usa la DESCRIZIONE MODELLO per capire il significato di ogni campo e cosa inserire.
    - Mantieni il testo nei campi il più breve possibile.

    Regole:
    - Se la RISPOSTA è esattamente "Non ho trovato la risposta nei documenti forniti" oppure non copre un campo → metti "".
    - Sì/No: usa "Sì" o "No".
    - Elenchi: elementi separati da virgole, senza punto finale.
    - Numeri e date: riporta esattamente il testo così come compare nella RISPOSTA, senza convertirli né modificarli (es. “duecento” resta “duecento”).
    - Non aggiungere informazioni non presenti nella RISPOSTA.
    - Output: SOLO il JSON del modello richiesto; nessun testo extra.

    HUMAN:

    DOMANDA:
    {question}

    RISPOSTA:
    {answer}

    Compila il modello {sub_module} usando SOLO la RISPOSTA e seguendo la DESCRIZIONE MODELLO.                          
"""
)


def return_keys_description_schema(schema: BaseModel) -> str:
    output = {}
    for k, v in schema.model_json_schema()["properties"].items():
        output[k] = {"descrizione": v["description"], "type": v["type"]}

    return json.dumps(output, indent=4, ensure_ascii=False)


EXTRACTOR_SYSTEM_PROMPT_V1 = textwrap.dedent(
    """\
    SYSTEM:

        Sei un estrattore di informazioni. Il tuo unico compito è compilare un oggetto JSON.

        =========================================
        STRUTTURA JSON DA COMPILARE:
        =========================================
        {sub_module_description}
        =========================================

        REGOLE ASSOLUTE:
        1. Rispondi SOLO con un oggetto JSON valido. Zero testo prima o dopo.
        2. Il JSON deve contenere ESATTAMENTE le chiavi indicate nella STRUTTURA JSON DA COMPILARE.
        3. Compila ogni chiave usando SOLO le informazioni presenti nella RISPOSTA qui sotto.
        4. NON inventare, NON dedurre, NON usare conoscenze esterne.

        REGOLE DI COMPILAZIONE DEI VALORI:
        - Campo non trovato nella RISPOSTA → valore: ""
        - La RISPOSTA è "Non ho trovato la risposta nei documenti forniti" → tutti i valori: ""
        - Sì/No → usa esattamente: "Sì" oppure "No"
        - Elenchi → elementi separati da virgola, senza punto finale
        - Numeri e date → copia il testo esatto dalla RISPOSTA (es. "duecento" NON diventa "200")
        - Testo breve → usa il minimo di parole necessarie

        FORMATO DI OUTPUT OBBLIGATORIO:
        {
        "chiave_1": "valore estratto o stringa vuota",
        "chiave_2": "valore estratto o stringa vuota"
        }

    HUMAN:

        RISPOSTA DA ANALIZZARE:
        {answer}

        Compila il JSON con le chiavi della STRUTTURA JSON DA COMPILARE usando SOLO la RISPOSTA sopra.
        Scrivi SOLO il JSON. Nessun testo aggiuntivo.
"""
)


@retry(wait=wait_random_exponential(min=1, max=60))
def extract_with_GOOGLE_API(
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
    response = client.models.generate_content(
        model=os.environ.get(
            "EXTRACTOR__GEMINI_MODEL_ID", ""
        ),  # "gemma-4-26b-a4b-it", #"gemma-4-31b-it", #os.environ.get("EXTRACTOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0,  # type: ignore
        ),
    )
    logging.info("RESPONSE FROM GOOGLE API (inside function): %s", response.text)
    if response.text:
        clean_string = re.sub(
            r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE
        )
        response_json = json.loads(clean_string)
    else:
        logging.error("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(
            info_schema.model_json_schema()["properties"]
        )

    return response_json


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

    client = genai.Client(
        http_options=types.HttpOptions(retry_options=retry_options, timeout=120 * 1000)
    )

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = await client.aio.models.generate_content(
        model=os.environ.get("EXTRACTOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0,  # type: ignore
        ),
    )
    logging.info("RESPONSE FROM GOOGLE API (inside function): %s", response.text)
    if response.text:
        clean_string = re.sub(
            r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE
        )
        response_json = json.loads(clean_string)
    else:
        logging.exception("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(
            info_schema.model_json_schema()["properties"]
        )

    return response_json


def extract_with_Ollama(model: str, info_schema: BaseModel, prompt_content: str):
    llm_for_extraction = OllamaLLM(llm_model=model, temperature=0)

    response = llm_for_extraction.invoke(
        output_format="structured",
        info_schema=info_schema,
        memory=prompt_content,
        num_predict=64,
        temperature=0,
        cache=False,
    )

    return response.model_dump()  # type: ignore


async def aextract_with_Ollama(model: str, info_schema: BaseModel, prompt_content: str):
    llm_for_extraction = OllamaLLM(llm_model=model, temperature=0)

    response = await llm_for_extraction.ainvoke(
        output_format="structured",
        info_schema=info_schema,
        memory=prompt_content,
        num_predict=64,
        temperature=0,
        cache=False,
    )

    return response.model_dump()  # type: ignore


if __name__ == "__main__":
    module = TermineApprovazioneBilancio
    question = module.question
    # answer = "Il capitale sociale è di euro 25.000 (venticinquemila)"
    answer = """
    L'Assemblea deve essere convocata almeno una volta l'anno per
l'approvazione del bilancio, e
ntro centoventi giorni dalla chiusura
dell'esercizio sociale, oppure ove la società sia tenuta alla redazione del
bilancio consolidato ovvero quando lo richiedano particolari esigenze
relative alla struttura ed all’oggetto della società, entro centottanta giorni
dalla sopradetta chiusura; in questi casi gli amministratori segnalano nella
relazione prevista dall’art. 2428 del codice civile le ragioni della dilazione.
    """

    # prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{question}", question).replace("{answer}", answer).replace("{sub_module}", CapitaleSociale.model_json_schema()['title']).replace("{sub_module_description}", json.dumps(CapitaleSociale.model_json_schema()['properties'], indent=2, ensure_ascii=False))

    schema_description = return_keys_description_schema(module)  # type:ignore
    prompt_content = EXTRACTOR_SYSTEM_PROMPT_V1.replace("{answer}", answer).replace(
        "{sub_module_description}", schema_description
    )

    # output = extract_with_GOOGLE_API(prompt_content, module) # type: ignore
    # output = asyncio.run(aextract_with_GOOGLE_API(prompt_content, module)) # type: ignore
    # output = extract_with_Ollama("gemma4:e2b", module, prompt_content) # type: ignore
    output = asyncio.run(aextract_with_Ollama("gemma4:e2b", module, prompt_content))  # type: ignore

    print(json.dumps(output, indent=4, ensure_ascii=False))
    module.model_validate(output)
