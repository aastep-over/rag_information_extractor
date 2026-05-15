from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tqdm import tqdm

# python native
from pathlib import Path
from typing import Dict, Type, List, Tuple
import json
import textwrap
import time
import re
import asyncio
import aiofiles
import os
import logging
from dotenv import load_dotenv

load_dotenv("../.env")

# from other modules
from rag_info_extractor.info_schema.utils import (
    formatted_word_to_number,
    load_classes_from_path,
    group_classes_by_module,
    return_default_json,
    return_keys_description_schema,
)
from rag_info_extractor.rag_pipeline import RAGPipeline
from rag_info_extractor.utils.llm_connector import OllamaLLM

# GEMINI API libraries
from google import genai
from tenacity import retry, wait_random_exponential
from google.genai import types

logger = logging.getLogger(__name__)

## ---------------------------------------------- ******* ------------------------------------------
## ---------------------------------------------- HELPERS ------------------------------------------

# Define Extracter Prompt and LLM

EXTRACTOR_SYSTEM_PROMPT = textwrap.dedent(
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

-----

HUMAN:

RISPOSTA DA ANALIZZARE:
{answer}

Compila il JSON con le chiavi della STRUTTURA JSON DA COMPILARE usando SOLO la RISPOSTA sopra.
Scrivi SOLO il JSON. Nessun testo aggiuntivo.
"""
)

# Define Post-Processing functions for extractions
POST_FUNCS = {
    "formatted_word_to_number": formatted_word_to_number,  # type:ignore (from utility functions)
}


# =======================================
# Extract with GOOGLE API
# =======================================
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
    logger.info("RESPONSE FROM GOOGLE API (inside function): %s", response.text)
    if response.text:
        clean_string = re.sub(
            r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE
        )
        response_json = json.loads(clean_string)
    else:
        logger.error("ERROR!NO RESPONSE FROM GOOGLE API")
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

    client = genai.Client(
        http_options=types.HttpOptions(retry_options=retry_options, timeout=120 * 1000)
    )

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = await client.aio.models.generate_content(
        model=os.environ.get("EXTRACTOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0,  # type: ignore
            # response_json_schema=info_schema.model_json_schema()['properties'] # type: ignore
        ),
    )

    if response.text:
        clean_string = re.sub(
            r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE
        )
        response_json = json.loads(clean_string)
    else:
        logger.exception("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(
            info_schema.model_json_schema()["properties"]
        )

    return response_json


## ---------------------------------------------- ******* ------------------------------------------
## ---------------------------------------------- EXTRACTI INFO CLASS  ------------------------------------------


class ExtractInfo:
    """Extract information from Schema"""

    def __init__(
        self,
        llm: OllamaLLM,
        extractor_graph: RAGPipeline,
        nome_azienda: str,
        sub_modules: List = [],
        use_google_api: bool = False,
    ) -> None:

        self.llm = llm
        self.sub_modules = sub_modules
        self.extractor_graph = extractor_graph
        self.nome_azienda = nome_azienda
        self.use_google_api = use_google_api

        self.out = {}

        self.ori_contexts = {}  ###
        self.re_ranked_contexts = {}  ###
        self.rag_qa = (
            {}
        )  ### # Stores question and answer by the rag pipeline for each submodule
        self.run_times = {}  ### # Store run time for each submodule
        self.ori_contexts_texts = {}  ###
        self.re_ranked_contexts_texts = {}  ###
        self.optimized_query = {}  ###

    def extract_sub_module(self, question: str, answer: str, sub_module):
        """
        Step 2: prende la DOMANDA e la RISPOSTA (testo generato dallo step 1) e
        compila le chiavi predefinite esclusivamente a partire dalla RISPOSTA.
        """
        logger.info(
            "\n --------------- NODE: __extract_info__ ------------------------\n"
        )  ###

        schema_description = return_keys_description_schema(sub_module)  # type:ignore
        prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{answer}", answer).replace(
            "{sub_module_description}", schema_description
        )

        if self.use_google_api:
            response = extract_with_GOOGLE_API(
                prompt_content=prompt_content, info_schema=sub_module
            )
            logger.info("RESPONSE FROM GOOGLE API: %s", response)
            result = sub_module.model_validate(response)
        else:
            result = self.llm.invoke(
                output_format="structured",
                info_schema=sub_module,
                memory=prompt_content,
                num_predict=64,
                temperature=0,
                cache=False,
            )  # type: ignore

        # Apply any required post-processing functions
        data = result.model_dump()  # type: ignore
        for func_name, field_names in getattr(
            sub_module, "post_process_func_var", {}
        ).items():
            fn = POST_FUNCS.get(func_name)
            if not fn:
                continue
            for field_name in field_names:
                value = data.get(field_name, "")
                data[field_name] = fn(value)

        return sub_module(**data)

    def extract_info(self):
        for m in self.sub_modules:
            logger.info(f"## \t Extracting info for moudle {m}: ")

            # Re-set time
            self.extractor_graph.reset_latency()

            logger.debug("Nome Azienda in : %s", self.nome_azienda.upper())  ###
            q = m.question + f" Nome della società: {self.nome_azienda.upper()}"
            answer = self.extractor_graph.get_response(q)
            time_consumed = self.extractor_graph.latency

            t1 = time.time()

            # structure answer only if llm found the response
            if answer not in ("", "Non ho trovato la risposta nei documenti forniti"):
                formatted_output = self.extract_sub_module(
                    m.question, answer, m
                ).model_dump()
            else:
                formatted_output = {}
                for k, v in m.model_json_schema()["properties"].items():
                    formatted_output[k] = v["default"]

            time_consumed["extract_sub_module"] = f"{time.time() - t1:.3f} s"

            name_sub_module: str = m.model_json_schema()["title"]
            self.out[name_sub_module] = formatted_output

            logger.debug("Module: %s", name_sub_module)  ###
            logger.debug("Module: %s", name_sub_module)  ###
            logger.debug("Question: \t %s", q)  ###
            logger.debug("Answer: \t %s", answer)  ###

            ## Save retrieved contexts and optimized query for testing
            self.optimized_query[name_sub_module] = (
                self.extractor_graph.optimized_query
            )  ###
            self.ori_contexts[name_sub_module] = (
                self.extractor_graph.retrieved_docs_ids
            )  ###
            self.ori_contexts_texts[name_sub_module] = (
                self.extractor_graph.retrieved_docs_texts
            )  ###
            try:
                self.re_ranked_contexts[name_sub_module] = (
                    self.extractor_graph.re_ranked_docs_ids
                )  ###
            except AttributeError as e:
                logger.error("re_ranked_docs_ids not found")
                logger.exception(e)
                self.re_ranked_contexts[name_sub_module] = {}  # []
            try:
                self.re_ranked_contexts_texts[name_sub_module] = (
                    self.extractor_graph.re_ranked_docs_texts
                )  ###
            except AttributeError as e:
                logger.error("re_ranked_docs_texts not found")
                logger.exception(e)
                self.re_ranked_contexts_texts[name_sub_module] = {}  ###

            self.rag_qa[name_sub_module] = {"Q": q, "A": answer}  ###

            time_consumed["overall"] = (
                f"{float(time_consumed.get('overall', '0 s')[:-2]) + float(time_consumed.get('extract_sub_module', '0 s')[:-2])} s"  ###
            )
            self.run_times[name_sub_module] = time_consumed  ###

            logger.debug("Time consumed on %s: %s", name_sub_module, time_consumed)  ###

    # ============== Async versions ========================
    async def aextract_sub_module(self, question: str, answer: str, sub_module):
        """Async implementation of extract_sub_module"""
        logger.info(
            "\n --------------- NODE: (async) __extract_info__ ------------------------\n"
        )  ###

        schema_description = return_keys_description_schema(sub_module)  # type:ignore
        prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{answer}", answer).replace(
            "{sub_module_description}", schema_description
        )
        if self.use_google_api:
            response = await aextract_with_GOOGLE_API(
                prompt_content=prompt_content, info_schema=sub_module
            )
            result = sub_module.model_validate(response)
        else:
            result = await self.llm.ainvoke(
                output_format="structured",
                info_schema=sub_module,
                memory=prompt_content,
                num_predict=64,
                temperature=0,
                cache=False,
            )  # type: ignore

        # Apply any required post-processing functions
        data = result.model_dump()  # type: ignore
        post_funcs_fields = getattr(sub_module, "post_process_func_var", {})

        async def _post_process_wrapper(func_name, field_name):
            fn = POST_FUNCS.get(func_name)
            if fn:
                # 1. Extract the actual value from data (defaulting to "" like the sync version)
                value = data.get(field_name, "")

                # 2. Push heavy function task to thread, passing the VALUE
                processed_value = await asyncio.to_thread(fn, value)

                # 3. Reassign the processed value back to the data dictionary
                data[field_name] = processed_value

        # Corrected list comprehension for tasks
        tasks = [
            _post_process_wrapper(func_name, field_name)
            for func_name, field_names in post_funcs_fields.items()
            for field_name in field_names  # Iterate over the specific fields assigned to this function
        ]

        logger.info("Running Post Processing Functions")
        if tasks:
            await asyncio.gather(*tasks)

        return sub_module(**data)

    async def aextract_info(self):

        async def _extract_info_wrapper(m):
            logger.info(f"## \t Extracting info for moudle {m}: ")

            # Re-set time
            self.extractor_graph.reset_latency()
            logger.debug("Nome Azienda in : %s", self.nome_azienda.upper())  ###
            q = m.question + f" Nome della società: {self.nome_azienda.upper()}"
            answer = await self.extractor_graph.aget_response(q)
            time_consumed = self.extractor_graph.latency

            t1 = time.perf_counter()

            # structure answer only if llm found the response
            if answer not in ("", "Non ho trovato la risposta nei documenti forniti"):
                formatted_output = await self.aextract_sub_module(m.question, answer, m)
                formatted_output = formatted_output.model_dump()
            else:
                formatted_output = {}
                for k, v in m.model_json_schema()["properties"].items():
                    formatted_output[k] = v["default"]

            time_consumed["extract_sub_module"] = f"{time.perf_counter() - t1:.3f} s"

            name_sub_module: str = m.model_json_schema()["title"]
            self.out[name_sub_module] = formatted_output

            logger.debug("Module: %s", name_sub_module)  ###
            logger.debug("Question: \t %s", q)  ###
            # logger.debug("INPUT TO RAG: %s", q_new
            logger.debug("Answer: \t %s", answer)  ###

            ## Save retrieved contexts and optimized query for testing
            self.optimized_query[name_sub_module] = (
                self.extractor_graph.optimized_query
            )  ###
            self.ori_contexts[name_sub_module] = (
                self.extractor_graph.retrieved_docs_ids
            )  ###
            self.ori_contexts_texts[name_sub_module] = (
                self.extractor_graph.retrieved_docs_texts
            )  ###
            try:
                self.re_ranked_contexts[name_sub_module] = (
                    self.extractor_graph.re_ranked_docs_ids
                )  ###
            except AttributeError as e:
                logger.error("re_ranked_docs_ids not found")
                logger.exception(e)
                self.re_ranked_contexts[name_sub_module] = {}  # []
            try:
                self.re_ranked_contexts_texts[name_sub_module] = (
                    self.extractor_graph.re_ranked_docs_texts
                )  ###
            except AttributeError as e:
                logger.error("re_ranked_docs_texts not found")
                logger.exception(e)
                self.re_ranked_contexts_texts[name_sub_module] = {}  ###

            self.rag_qa[name_sub_module] = {"Q": q, "A": answer}  ###

            time_consumed["overall"] = (
                f"{float(time_consumed.get('overall', '0 s')[:-2]) + float(time_consumed.get('extract_sub_module', '0 s')[:-2])} s"  ###
            )
            self.run_times[name_sub_module] = time_consumed  ###

            logger.debug("Time consumed on %s: %s", name_sub_module, time_consumed)

        await asyncio.gather(*[_extract_info_wrapper(m) for m in self.sub_modules])

    @property
    def output(self):
        return self.out


# ============================== MAIN FUNCTION ============================================


def extract_and_save_all_info(
    rag_pipeline: RAGPipeline,
    nome_delle_aziende: List[Tuple[str, str]],
    llm_json: OllamaLLM,
    save_dir: str,
    use_google_api: bool = False,
) -> None:

    info_schema_classes = load_classes_from_path(
        "../src/rag_info_extractor/info_schema/schemas"
    )
    info_to_extract_classes = group_classes_by_module(info_schema_classes)
    os.makedirs(save_dir, exist_ok=True)

    info_extracted = {}
    for azienda in tqdm(
        nome_delle_aziende, desc="Extracting info. for the azienda"
    ):  # tqdm(docs_paths, desc="Completed")
        # Save info for each Azienda
        info_per_azienda = {}
        logger.info(f"Extracting for Azienda: {azienda}")

        for info_group_name, info_group_modules in tqdm(
            info_to_extract_classes.items(), desc="Extracting information"
        ):

            logger.info(f"\t Extracting info about {info_group_name} ")
            # Define object for each information class
            extractor_obj = ExtractInfo(
                llm=llm_json,
                extractor_graph=rag_pipeline,
                nome_azienda=azienda[0],
                sub_modules=info_group_modules,
                use_google_api=use_google_api,
            )

            # Extract infromation and save it
            extractor_obj.extract_info()
            info = extractor_obj.output
            info_per_azienda[info_group_name] = {
                "output": info,
                "retrieved_docs": extractor_obj.ori_contexts,
                "re_ranked_docs": extractor_obj.re_ranked_contexts,
                "retrieved_docs_texts": extractor_obj.ori_contexts_texts,
                "re_ranked_docs_texts": extractor_obj.re_ranked_contexts_texts,
                "rag_qa": extractor_obj.rag_qa,
                "run_times": extractor_obj.run_times,
                "optimized_query": extractor_obj.optimized_query,
            }
            logger.info(f'{"-"*40} xxxxx {"-"*40}')

            # save info for all aziende
            info_extracted[azienda[0]] = info_per_azienda
            logger.info(f"\t Information extracted for the group: {info_group_name}")
            # save check point (for current group and current azienda)
            with open(f"{save_dir}/last_run.json", "w", encoding="utf-8") as f:
                json.dump(info_extracted, f, indent=4, ensure_ascii=False)

        # save info for all aziende
        info_extracted[azienda[0]] = info_per_azienda
        logger.info(f'{"="*40} XXX {"="*40}')
        logger.info(f"Information extracted for the azienda: {azienda}.")
        # save checkpoint (for all groups for current azienda)
        with open(f"{save_dir}/last_run.json", "w", encoding="utf-8") as f:
            json.dump(info_extracted, f, indent=4, ensure_ascii=False)

    # save final checkpoint
    with open(f"{save_dir}/last_run.json", "w", encoding="utf-8") as f:
        json.dump(info_extracted, f, indent=4, ensure_ascii=False)

    # write final result to pred.json
    filename = f"{save_dir}/pred.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(info_extracted, f, indent=4, ensure_ascii=False)


async def aextract_and_save_all_info(
    rag_pipeline: RAGPipeline,
    nome_delle_aziende: List[Tuple[str, str]],
    llm_json: OllamaLLM,
    save_dir: str,
    use_google_api: bool = False,
) -> None:

    info_schema_classes = load_classes_from_path(
        "../src/rag_info_extractor/info_schema/schemas"
    )  # loader.exec_module(module) blocks from using 'load_classes_from_path' async
    info_to_extract_classes = group_classes_by_module(info_schema_classes)
    os.makedirs(save_dir, exist_ok=True)
    # Create last_runs dir
    last_runs_dir = Path(f"{save_dir}/last_runs")
    last_runs_dir.mkdir(parents=True, exist_ok=True)

    # ================================= Helper Functions ============================================
    async def _save_intermediate_json(info_extracted: dict, filename: str):
        # save check point (for current group and current azienda)
        json_data = json.dumps(
            info_extracted, indent=4, ensure_ascii=False
        )  # json.dump is synchronous

        async with aiofiles.open(last_runs_dir / filename, "w", encoding="utf-8") as f:
            await f.write(json_data)

    async def _extract_group(group_name: str, azienda: str) -> Tuple[str, dict]:
        logger.info(f"\t Extracting info about {group_name} ")

        # Define ExtractInfo object for each information class
        extractor_obj = ExtractInfo(
            llm=llm_json,
            extractor_graph=rag_pipeline,
            nome_azienda=azienda,
            sub_modules=info_to_extract_classes[group_name],
            use_google_api=use_google_api,
        )

        # Extract infromation and save it
        await extractor_obj.aextract_info()
        info = extractor_obj.output
        logger.debug("\t %s xxxxx %s", "=" * 40, "=" * 40)

        return (
            group_name,
            {
                "output": info,
                "retrieved_docs": extractor_obj.ori_contexts,
                "re_ranked_docs": extractor_obj.re_ranked_contexts,
                "retrieved_docs_texts": extractor_obj.ori_contexts_texts,
                "re_ranked_docs_texts": extractor_obj.re_ranked_contexts_texts,
                "rag_qa": extractor_obj.rag_qa,
                "run_times": extractor_obj.run_times,
                "optimized_query": extractor_obj.optimized_query,
            },
        )

    async def _extract_per_azienda(azienda: str):
        logger.info("Extracting for Azienda: %s", azienda)

        # Run extraction async
        group_tasks = [
            _extract_group(group, azienda) for group in info_to_extract_classes.keys()
        ]
        per_azienda_results = await asyncio.gather(*group_tasks)

        # save json per azienda
        info_per_azienda = dict(per_azienda_results)
        await _save_intermediate_json(info_per_azienda, f"{azienda}.json")

        logger.info("%s XXX %s", "=" * 40, "=" * 40)

    # ============================================ XXX ===================================

    azienda_tasks = [
        _extract_per_azienda(azienda) for azienda, f_name in nome_delle_aziende
    ]
    await asyncio.gather(*azienda_tasks)

    # Use per azienda files to create a pred.json
    combined_data = {}

    async def _load_all_aziende_jsons(dir: Path):
        # Helper to load one azienda json
        async def __load_one_azienda_json(filename: Path):
            async with aiofiles.open(filename, "r", encoding="utf-8") as f:
                data = await f.read()
            combined_data[filename.stem] = json.loads(data)

        tasks = [__load_one_azienda_json(f) for f in last_runs_dir.iterdir()]
        await asyncio.gather(*tasks)

    await _load_all_aziende_jsons(last_runs_dir)

    with open(f"{save_dir}/pred.json", "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)


# TODO: Fix async pipeline state handling for per-query metadata.
# Currently, fields such as `optimized_queries`, `re_ranked_docs`,
# and `retrieved_docs` are being overwritten/shared across async calls,
# causing all queries/modules to reference the results of the last completed task
# instead of maintaining isolated state per question/query/module.
