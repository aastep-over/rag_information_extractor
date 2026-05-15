from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

# Python native
from typing import List, Dict, Optional, Tuple
import re
import textwrap
import os
from dotenv import load_dotenv

# from other modules
from rag_info_extractor.utils.llm_connector import OllamaLLM

# Logging
import logging
logger = logging.getLogger(__name__)

# Google API related
from google import genai
from tenacity import retry, wait_random_exponential
from google.genai import types


# Define System prompt for generation
SYSTEM_PROMPT = textwrap.dedent("""\
        SYSTEM:
        Sei un analista di statuti societari.

        ISTRUZIONI:
        - Ti verranno dati più CHUNK separati da "||".
        - Leggi ogni CHUNK separatamente per trovare la risposta alla domanda.
        - Usa solo le informazioni esplicite nei chunk.
        - Se più chunk contengono parti utili, combina solo ciò che serve in una frase chiara e breve.
        - Se nessuno contiene la risposta, scrivi esattamente: "Non ho trovato la risposta nei documenti forniti".
        - Rispondi sempre in italiano, senza elenco puntato o testo extra.

        HUMAN:
        {additional_prompt}

        CONTESTO:
        {context}

        DOMANDA:
        {question}                          
    """)

# =======================================
#  Answer with GOOGLE API
# =======================================
@retry(wait=wait_random_exponential(min=1, max=60))
def answer_with_GOOGLE_API(
    prompt_content: str
):

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    retry_options = types.HttpRetryOptions(
        initial_delay=2.0,      
        max_delay=60.0,         
        exp_base=2.0,         
        attempts=10,             
        http_status_codes=[408, 429, 500, 502, 503, 504]
    )

    client = genai.Client(
        http_options=types.HttpOptions(
            retry_options=retry_options
        )
    )

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = client.models.generate_content(
        model=os.environ.get("GENERATOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )

    return response

# Async version
@retry(wait=wait_random_exponential(min=1, max=60))
async def aanswer_with_GOOGLE_API(
    prompt_content: str,
):

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    retry_options = types.HttpRetryOptions(
        initial_delay=2.0,      
        max_delay=60.0,         
        exp_base=2.0,         
        attempts=10,             
        http_status_codes=[408, 429, 500, 502, 503, 504]
    )

    client = genai.Client(
        http_options=types.HttpOptions(
            retry_options=retry_options
        )
    )

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = await client.aio.models.generate_content(
        model=os.environ.get("GENERATOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0,
        )
    )

    return response



# ======= MAIN function ===============

def generate(
    question: str,
    contexts: List[Document],
    llm: OllamaLLM,
    additional_prompt: str = "",
    contexts_sep: str = "||",
    use_google_api: bool = False,

) -> str:
    
    """Generate answer to query/question using the context retrieved by the retriever"""

    logger.info("\n --------------- NODE: __generate__ ------------------------\n")###
    docs_content = contexts_sep.join(doc.page_content for doc in contexts) # legacy joiner: "\n\n"
    
    # Remove name of società to avoid confusion for llm
    contain_name = re.findall(r"(.*) Nome della società: (.*)", question) 
    if contain_name:
        q = contain_name[0][0]
        azienda_name = contain_name[0][1]
    else:
        q = question

    # Update prompt with additional_instruction, context and question
    prompt_content = SYSTEM_PROMPT.replace("{additional_prompt}", additional_prompt)
    prompt_content = prompt_content.replace("{context}", docs_content)
    prompt_content = prompt_content.replace("{question}", q)
    
    if use_google_api:
        if contain_name:
            prompt_content = prompt_content.replace(azienda_name, "<nome_azienda>") # response.text.strip()
        response_api = answer_with_GOOGLE_API(prompt_content)
        ai_answer: AIMessage = AIMessage(content=response_api.text) # type: ignore
    else:
        ai_answer: AIMessage = llm.invoke(
            output_format = "text",
            memory = prompt_content,
            num_predict = 500,
            temperature = 0
        ) # type: ignore

    if isinstance(ai_answer.content, str):
        answer = ai_answer.content.strip() 
    else:
        answer = "Non ho trovato la risposta"

    return answer

# Async version
async def agenerate(
    question: str,
    contexts: List[Document],
    llm: OllamaLLM,
    additional_prompt: str = "",
    contexts_sep: str = "||",
    use_google_api: bool = False,
    
) -> str:

    """Async implementation of generate"""

    logger.info("\n --------------- NODE: (async) __generate__ ------------------------\n")###
    docs_content = contexts_sep.join(doc.page_content for doc in contexts) # legacy joiner: "\n\n"
    
    # Remove name of società to avoid confusion for llm
    contain_name = re.findall(r"(.*) Nome della società: (.*)", question) 
    if contain_name:
        q = contain_name[0][0]
        azienda_name = contain_name[0][1]
    else:
        q = question
    
    # Update prompt with additional_instruction, context and question
    prompt_content = SYSTEM_PROMPT.replace("{additional_prompt}", additional_prompt)
    prompt_content = prompt_content.replace("{context}", docs_content)
    prompt_content = prompt_content.replace("{question}", q)

    if use_google_api:
        if contain_name:
            prompt_content = prompt_content.replace(azienda_name, "<nome_azienda>") # response.text.strip()
        response_api = await aanswer_with_GOOGLE_API(prompt_content)
        ai_answer: AIMessage = AIMessage(content=response_api.text) # type: ignore
    else:
        ai_answer: AIMessage = await llm.ainvoke(
            output_format = "text",
            memory = prompt_content,
            num_predict = 500,
            temperature = 0
        ) # type: ignore

    if isinstance(ai_answer.content, str):
        answer = ai_answer.content.strip() 
    else:
        answer = "Non ho trovato la risposta"

    return answer

