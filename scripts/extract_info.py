from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tqdm import tqdm


# python native
from pathlib import Path
from typing import Dict, Type, List, Tuple
import json
import textwrap
import datetime
import re
import asyncio
import aiofiles

# from other modules
from rag_info_extractor.info_schema.utils import formatted_word_to_number, load_classes_from_path, group_classes_by_module, return_default_json, return_keys_description_schema
from rag_pipeline import RAGPipeline #rag_information_extractor/scripts/rag_pipeline.py
from rag_info_extractor.utils.llm_connector import OllamaLLM

# GEMINI API prova
from google import genai
from tenacity import retry, wait_random_exponential
from google.genai import types

## ---------------------------------------------- ******* ------------------------------------------
## ---------------------------------------------- HELPERS ------------------------------------------

# TODO: Test V1 for both with api and local ollama to and put v0 to legacy if works
# Define Extracter Prompt and LLM 
EXTRACTOR_SYSTEM_PROMPT = textwrap.dedent("""\
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
""")

# Define Post-Processing functions for extractions
POST_FUNCS = {
    "formatted_word_to_number": formatted_word_to_number, # type:ignore (from utility functions)
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
        http_status_codes=[408, 429, 500, 502, 503, 504]
    )

    client = genai.Client(
        http_options=types.HttpOptions(
            retry_options=retry_options
        )
    )

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = client.models.generate_content(
        model= os.environ.get("EXTRACTOR__GEMINI_MODEL_ID", ""),#"gemma-4-26b-a4b-it", #"gemma-4-31b-it", #os.environ.get("EXTRACTOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0, # type: ignore
        )
    )
    logger.info("RESPONSE FROM GOOGLE API (inside function): %s", response.text)
    if response.text:
        clean_string = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        response_json = json.loads(clean_string)
    else:
        logger.error("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(info_schema.model_json_schema()['properties'])

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
        http_status_codes=[408, 429, 500, 502, 503, 504]
    )

    client = genai.Client(
        http_options=types.HttpOptions(
            retry_options=retry_options,
            timeout=120 * 1000
        )
    )

    # check available models on https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1&hl=it
    response = await client.aio.models.generate_content(
        model=os.environ.get("EXTRACTOR__GEMINI_MODEL_ID", ""),
        contents=prompt_content,
        config=types.GenerateContentConfig(
            temperature=0.0, # type: ignore
            # response_json_schema=info_schema.model_json_schema()['properties'] # type: ignore
        )
    )

    if response.text:
        clean_string = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        response_json = json.loads(clean_string)
    else:
        logger.exception("ERROR!NO RESPONSE FROM GOOGLE API")
        response_json = return_default_json(info_schema.model_json_schema()['properties'])

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
        optimized_query_per_group: dict = {}, 
        use_google_api: bool = True, 

    ) -> None:
        
        self.llm = llm
        self.sub_modules = sub_modules
        self.extractor_graph = extractor_graph
        self.nome_azienda = nome_azienda
        self.optimized_query_per_group = optimized_query_per_group # TODO: REMOVE AFTER TESTING
        self.use_google_api = use_google_api # TODO: REMOVE AFTER TESTING

        self.out = {}

        self.ori_contexts = {}###
        self.re_ranked_contexts = {}###
        self.rag_qa = {}### # Stores question and answer by the rag pipeline for each submodule
        self.run_times = {}### # Store run time for each submodule
        self.ori_contexts_texts = {}###
        self.re_ranked_contexts_texts = {}###
        self.optimized_query = {}###
    

    def extract_sub_module(self, question: str, answer: str, sub_module):
        """
        Step 2: prende la DOMANDA e la RISPOSTA (testo generato dallo step 1) e
        compila le chiavi predefinite esclusivamente a partire dalla RISPOSTA.
        """
        logger.info("\n --------------- NODE: __extract_info__ ------------------------\n")###

        prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{question}", question).replace("{answer}", answer).replace("{sub_module}", sub_module.model_json_schema()['title']).replace("{sub_module_description}", json.dumps(sub_module.model_json_schema()['properties'], indent=2, ensure_ascii=False))

        if self.use_google_api:
            response = extract_with_GOOGLE_API(
                prompt_content = prompt_content,
                info_schema = sub_module
            )
            logger.info("RESPONSE FROM GOOGLE API: %s", response)
            result = sub_module.model_validate(response)
        else:
            result = self.llm.invoke(
                output_format = "structured",
                info_schema = sub_module,
                memory = prompt_content,
                num_predict = 64,
                temperature = 0,
                cache = False
            ) # type: ignore

        # Apply any required post-processing functions
        data = result.model_dump() # type: ignore
        for func_name, field_names in getattr(sub_module, "post_process_func_var", {}).items():
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

            logger.debug('Nome Azienda in : %s', self.nome_azienda.upper())### 
            q = m.question + f" Nome della società: {self.nome_azienda.upper()}" 
            # q_new = m.question + f" QUERY: {self.optimized_query_per_group[m.model_json_schema()['title']]['query']}" + f" Nome della società: {self.nome_azienda}"  # TODO: REMOVE AFTER TESTING
            answer = self.extractor_graph.get_response(q) 
            time_consumed = self.extractor_graph.latency
            
            t1 = time.time()

            # structure answer only if llm found the response
            if answer not in ("", "Non ho trovato la risposta nei documenti forniti"):
                formatted_output = self.extract_sub_module(m.question, answer, m).model_dump()
            else:
                formatted_output = {}
                for k, v in m.model_json_schema()['properties'].items():
                    formatted_output[k] = v['default']


            time_consumed["extract_sub_module"] = f"{time.time() - t1:.3f} s"

            name_sub_module: str = m.model_json_schema()['title']
            self.out[name_sub_module] = formatted_output


            logger.debug("Module: %s", name_sub_module)###
            logger.debug("Module: %s", name_sub_module)###
            logger.debug("Question: \t %s", q)###
            # logger.debug("INPUT TO RAG: %s", q_new)
            logger.debug("Answer: \t %s", answer)###


            ## Save retrieved contexts and optimized query for testing
            self.optimized_query[name_sub_module] = self.extractor_graph.optimized_query###
            self.ori_contexts[name_sub_module] = self.extractor_graph.retrieved_docs_ids###
            self.ori_contexts_texts[name_sub_module] = self.extractor_graph.retrieved_docs_texts###
            try:
                self.re_ranked_contexts[name_sub_module] = self.extractor_graph.re_ranked_docs_ids###
            except AttributeError as e:
                logger.error("re_ranked_docs_ids not found")
                logger.exception(e)
                self.re_ranked_contexts[name_sub_module] = {}#[]
            try:
                self.re_ranked_contexts_texts[name_sub_module] = self.extractor_graph.re_ranked_docs_texts###
            except AttributeError as e:
                logger.error("re_ranked_docs_texts not found")
                logger.exception(e)
                self.re_ranked_contexts_texts[name_sub_module] = {}###

            self.rag_qa[name_sub_module] = {"Q": q, "A": answer}###

            time_consumed['overall'] = f"{float(time_consumed.get('overall', '0 s')[:-2]) + float(time_consumed.get('extract_sub_module', '0 s')[:-2])} s"###
            self.run_times[name_sub_module] = time_consumed###

            logger.debug("Time consumed on %s: %s", name_sub_module, time_consumed)###

    # ============== Async versions ========================
    async def aextract_sub_module(self, question: str, answer: str, sub_module):
        """Async implementation of extract_sub_module"""
        logger.info("\n --------------- NODE: __extract_info__ ------------------------\n")###

        prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{question}", question).replace("{answer}", answer).replace("{sub_module}", sub_module.model_json_schema()['title']).replace("{sub_module_description}", json.dumps(sub_module.model_json_schema()['properties'], indent=2, ensure_ascii=False))
        if self.use_google_api:
            response = await aextract_with_GOOGLE_API(
                prompt_content = prompt_content,
                info_schema = sub_module
            )
            print(response)
            result = sub_module.model_validate(response)
        else:
            result = await self.llm.ainvoke(
                output_format = "structured",
                info_schema = sub_module,
                memory = prompt_content,
                num_predict = 64,
                temperature = 0,
                cache = False
            ) # type: ignore

        # Apply any required post-processing functions
        data = result.model_dump() # type: ignore
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
            for field_name in field_names # Iterate over the specific fields assigned to this function
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
            logger.debug("Nome Azienda in : %s", self.nome_azienda.upper())### 
            q = m.question + f" Nome della società: {self.nome_azienda.upper()}"
            # q_new = m.question + f" QUERY: {self.optimized_query_per_group[m.model_json_schema()['title']]['query']}" + f" Nome della società: {self.nome_azienda}"  # TODO: REMOVE AFTER TESTING
            answer = await self.extractor_graph.aget_response(q)
            time_consumed = self.extractor_graph.latency

            t1 = time.perf_counter()

            # structure answer only if llm found the response
            if answer not in ("", "Non ho trovato la risposta nei documenti forniti"):
                formatted_output = await self.aextract_sub_module(m.question, answer, m)
                formatted_output = formatted_output.model_dump()
            else:
                formatted_output = {}
                for k, v in m.model_json_schema()['properties'].items():
                    formatted_output[k] = v['default']


            time_consumed["extract_sub_module"] = f"{time.perf_counter() - t1:.3f} s"

            name_sub_module: str = m.model_json_schema()['title']
            self.out[name_sub_module] = formatted_output


            logger.debug("Module: %s", name_sub_module)###
            logger.debug("Question: \t %s", q)###
            # logger.debug("INPUT TO RAG: %s", q_new
            logger.debug("Answer: \t %s", answer)###


            ## Save retrieved contexts and optimized query for testing
            self.optimized_query[name_sub_module] = self.extractor_graph.optimized_query###
            self.ori_contexts[name_sub_module] = self.extractor_graph.retrieved_docs_ids###
            self.ori_contexts_texts[name_sub_module] = self.extractor_graph.retrieved_docs_texts###
            try:
                self.re_ranked_contexts[name_sub_module] = self.extractor_graph.re_ranked_docs_ids###
            except AttributeError as e:
                logger.error("re_ranked_docs_ids not found")
                logger.exception(e)
                self.re_ranked_contexts[name_sub_module] = {}#[]
            try:
                self.re_ranked_contexts_texts[name_sub_module] = self.extractor_graph.re_ranked_docs_texts###
            except AttributeError as e:
                logger.error("re_ranked_docs_texts not found")
                logger.exception(e)
                self.re_ranked_contexts_texts[name_sub_module] = {}###

            self.rag_qa[name_sub_module] = {"Q": q, "A": answer}###

            time_consumed['overall'] = f"{float(time_consumed.get('overall', '0 s')[:-2]) + float(time_consumed.get('extract_sub_module', '0 s')[:-2])} s"###
            self.run_times[name_sub_module] = time_consumed###

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
    optimized_queries: dict = {} 
) -> None:
    
    info_schema_classes = load_classes_from_path("../src/rag_info_extractor/info_schema/schemas") 
    info_to_extract_classes = group_classes_by_module(info_schema_classes)
    os.makedirs(save_dir, exist_ok=True)

    info_extracted = {}
    for azienda in tqdm(nome_delle_aziende, desc="Extracting info. for the azienda"): #tqdm(docs_paths, desc="Completed")
        # Save info for each Azienda
        info_per_azienda = {}
        logger.info(f"Extracting for Azienda: {azienda}")

        for info_group_name, info_group_modules in tqdm(info_to_extract_classes.items(), desc="Extracting information"):

            logger.info(f"\t Extracting info about {info_group_name} ") 
            # Define object for each information class
            extractor_obj = ExtractInfo(
                llm = llm_json,
                extractor_graph = rag_pipeline,
                nome_azienda = azienda[0],
                sub_modules = info_group_modules,
                # optimized_query_per_group = optimized_queries[azienda[0]][info_group_name] # TODO: REMOVE AFTER TESTING
            )
            
            # Extract infromation and save it
            extractor_obj.extract_info() # await extractor_obj.aextract_info()
            info = extractor_obj.output
            info_per_azienda[info_group_name] = {  
                "output": info,
                "retrieved_docs": extractor_obj.ori_contexts,
                "re_ranked_docs": extractor_obj.re_ranked_contexts,
                "retrieved_docs_texts": extractor_obj.ori_contexts_texts,
                "re_ranked_docs_texts": extractor_obj.re_ranked_contexts_texts,
                "rag_qa": extractor_obj.rag_qa,
                "run_times": extractor_obj.run_times,
                "optimized_query": extractor_obj.optimized_query
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
    optimized_queries: dict = {} 
) -> None:

    info_schema_classes = load_classes_from_path("../src/rag_info_extractor/info_schema/schemas") # loader.exec_module(module) blocks from using 'load_classes_from_path' async
    info_to_extract_classes = group_classes_by_module(info_schema_classes)
    os.makedirs(save_dir, exist_ok=True)
    # Create last_runs dir
    last_runs_dir = Path(f"{save_dir}/last_runs")
    last_runs_dir.mkdir(parents=True, exist_ok=True)

    # ================================= Helper Functions ============================================
    async def _save_intermediate_json(info_extracted: dict, filename: str):
        # save check point (for current group and current azienda)
        json_data = json.dumps(info_extracted, indent=4, ensure_ascii=False) # json.dump is synchronous

        async with aiofiles.open(last_runs_dir / filename, "w", encoding="utf-8") as f:
            await f.write(json_data)

    async def _extract_group(group_name: str, azienda: str) -> Tuple[str, dict]:
        logger.info(f"\t Extracting info about {group_name} ")

        # Define ExtractInfo object for each information class
        extractor_obj = ExtractInfo(
            llm = llm_json,
            extractor_graph = rag_pipeline,
            nome_azienda = azienda,
            sub_modules = info_to_extract_classes[group_name],
            # optimized_query_per_group = optimized_queries[azienda][group_name] # TODO: REMOVE AFTER TESTING
        )

        # Extract infromation and save it
        await extractor_obj.aextract_info()
        info = extractor_obj.output
        # logger.debug(f'\t {"-"*40} xxxxx {"-"*40}')
        logger.debug("\t %s xxxxx %s", "="*40, "="*40)

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
                "optimized_query": extractor_obj.optimized_query
            } 
        )
        
    async def _extract_per_azienda(azienda: str):
        logger.info("Extracting for Azienda: %s", azienda)

        # Run extraction async
        group_tasks = [_extract_group(group, azienda) for group in info_to_extract_classes.keys()]
        per_azienda_results = await asyncio.gather(*group_tasks)
        
        # save json per azienda
        info_per_azienda = dict(per_azienda_results)
        await _save_intermediate_json(info_per_azienda, azienda)

        # logger.info(f'{"="*40} XXX {"="*40}')
        logger.info('%s XXX %s', "="*40, "="*40)
    
    # ============================================ XXX ===================================

    azienda_tasks = [_extract_per_azienda(azienda) for azienda, f_name in nome_delle_aziende]
    await asyncio.gather(*azienda_tasks)
    
    # Use per azienda files to create a pred.json 
    combined_data = {}
    async def _load_one_azienda_json(filename: Path):
        async with aiofiles.open(filename, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        combined_data[filename.stem] = data

    asyncio.gather(*[_load_one_azienda_json(fname) for fname in last_runs_dir.glob("*.json")])

    with open(f"{save_dir}/pred.json", "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    
    from langchain_chroma import Chroma
    from langchain.storage import LocalFileStore
    from langchain_huggingface import HuggingFaceEmbeddings

    import os
    import yaml
    import time, datetime
    import argparse
    from dotenv import load_dotenv

    # logging relative
    import logging
    from rag_info_extractor.utils.common_logging import configure_logging
    from rag_info_extractor.utils.load_config import cfgs
    from rag_info_extractor.utils.embedder import HFEmbedder


    logger = logging.getLogger(__name__)


    t0 = time.time()

    # CONFIG File Settings
    cfgs = cfgs.get("args", {})

    LLM_MODEL = cfgs.get("LLM_MODEL")
    EXTRACTOR_LLM = cfgs.get("EXTRACTOR_LLM")

    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    USE_GOOGLE_API = cfgs.get("USE_GOOGLE_API")

    # Modify config setting from CLI for automated testing if needed
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM Model for RAG pipeline"
    )
    parser.add_argument(
        "--extractor-llm",
        type=str,
        help="LLM Model for RAG pipeline"
    )
    parser.add_argument(
        "--chunks-type",
        type=str,
        choices=["custom_chunks", "fixed_size_chunks", "semantic_chunks", "custom_chunks_2"],
        help="Chunking method used to extract context from",
    )

    args = parser.parse_args()
    if args.llm_model:
        LLM_MODEL = args.llm_model
        cfgs['LLM_MODEL'] = LLM_MODEL
    if args.extractor_llm:
        EXTRACTOR_LLM = args.extractor_llm
        cfgs['EXTRACTOR_LLM'] = LLM_MODEL
    if args.chunks_type:
        CHUNKS_TYPE = args.chunks_type
        cfgs['CHUNKS_TYPE'] = CHUNKS_TYPE 
    
    # Define Paths to larger(Parent) chunks, vectorDB (children chunks)
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) # f"../data/large_chunks_dbs/{DATASET_TYPE}/{CHUNKS_TYPE}"
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE) # f"../data/vector_dbs/{DATASET_TYPE}/{CHUNKS_TYPE}"
    # Ensure large chunks path and vector db exist
    if not os.path.exists(DOC_STORE_LARGE_CHUNKS_PATH) or not os.scandir(DOC_STORE_LARGE_CHUNKS_PATH):
        logger.exception("Large Chunks Path: %s  does not exist", DOC_STORE_LARGE_CHUNKS_PATH)
        raise FileNotFoundError("Large Chunks Path does not exist")
    if not os.path.exists(VECTOR_STORE_PATH) or not os.scandir(VECTOR_STORE_PATH): 
        logger.exception("Vector DB Path: %s  does not exist", VECTOR_STORE_PATH)
        raise FileNotFoundError("Vector DB Path does not exist")

    # Configure logging settings
    RUN_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    LOGDIR = os.path.join(BASE_DIR, "logs", "extract_info_py")
    os.makedirs(LOGDIR, exist_ok=True)
    configure_logging(
        default_level=logging.DEBUG if args.verbose else logging.INFO,
        logfile=os.path.join(LOGDIR, f"{RUN_TIME}.log")
    )

    # Outputs are saved to BASE_DIR/runs/...
    OUTPUT_SAVE_DIR = os.path.join(BASE_DIR, 'runs', DATASET_TYPE, CHUNKS_TYPE, f"run_{RUN_TIME}")
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
    logger.info("Output will be saved to %s", OUTPUT_SAVE_DIR)

    # Load Vector and Doc store
    embedding = HFEmbedder(normalize_embeddings=True)
    vector_store = Chroma(embedding_function=embedding,
                        persist_directory=VECTOR_STORE_PATH,
                        collection_name="pdf_chunks")
    retriever = vector_store.as_retriever(search_type="similarity",
                                        search_kwargs={'k': 8}) 

    # Get all azienda names in vector/doc store
    nome_delle_aziende = set((vector_store.get()['metadatas'][i].get('azienda'), vector_store.get()['metadatas'][i].get('filename')) for i in range(len(vector_store.get()['ids']))) 
    nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    nome_delle_aziende = [x for x in nome_delle_aziende if x[0] != "unicredit s.p.a."]
    azienda_name_records = [x[0] for x in nome_delle_aziende if x[0] != "unicredit s.p.a."]

    # nome_delle_aziende = [
    #     ('compagnie de participation hotelliere et touristique', '8049135570002.pdf')
    # ]
    # azienda_name_records = [
    #     'compagnie de participation hotelliere et touristique', 
    # ]

# ============================================ TODO: REMOVE THIS BLOCK AFTER TESTING ============================================

    # # Take optimized queries from best gemma run # 
    # pred_json_path = "D:/Documents/Italy/UNIPD/University Acadamico/TESI/project/rag_information_extractor/runs/TRAIN/custom_chunks_2/run_2026-04-02 21-43-00/pred.json"
    # TOTAL_AZIENDA = 15 # Train:15, Test:8
    # with open(pred_json_path, encoding="utf-8") as f:
    #     pred_json = json.load(f)

    # azienda_optmized_queries = {}
    # i = 0 # to ensure different names across different dbs
    # pred_json_counter = 0
    # for azienda, azienda_data in pred_json.items():
    #     if pred_json_counter < TOTAL_AZIENDA - len(azienda_name_records):
    #         pred_json_counter += 1
    #         continue
        
    #     azienda_optmized_queries[azienda_name_records[i]] = {}
    #     for group, group_data in azienda_data.items():
    #         azienda_optmized_queries[azienda_name_records[i]][group] = {}
    #         for sg, sg_data in group_data["optimized_query"].items():
    #             azienda_optmized_queries[azienda_name_records[i]][group][sg] = sg_data
    #     i += 1

    
# ============================================ XXX ============================================

    # Load the schemas of info to be etracted
    classes = load_classes_from_path(os.path.join(BASE_DIR, "src", "rag_info_extractor", "info_schema", "schemas"))

    # Load RAG pipeline
    logger.info("Initializing RAG Pipeline...") ###
    try: # TODO: test and remove from try and except block
        rag_obj = RAGPipeline(
            db_retriever = retriever, 
            azienda_name_records = azienda_name_records,
            llm_model = LLM_MODEL,
            doc_store_path = DOC_STORE_LARGE_CHUNKS_PATH,
            pages_joining_str = PAGES_JOINING_STR,
            use_google_api = USE_GOOGLE_API
        ) 
    except Exception as e:
        logger.error("Error initializing RAG Pipeline.")
        logger.exception(e)
    else:
        logger.info("Initialized RAG Pipeline.") ###
        # save the DAG flow for RAG nodes/Pipeline
        rag_obj.save_DAG_diagram(OUTPUT_SAVE_DIR)
        logger.info("The DAG for RAG Pipeline saved to %s", OUTPUT_SAVE_DIR) 
    

    # Define LLM for extractions
    llm_for_extraction = OllamaLLM(
        llm_model=EXTRACTOR_LLM,
        temperature=0
    )

    # Save config
    with open(os.path.join(OUTPUT_SAVE_DIR, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfgs, f)

    # Extract info
    extract_and_save_all_info(
        rag_pipeline = rag_obj,
        nome_delle_aziende = nome_delle_aziende,
        llm_json = llm_for_extraction,
        save_dir = OUTPUT_SAVE_DIR,
        # optimized_queries = azienda_optmized_queries # TODO: REMOVE AFTER TESTING
    )
    # asyncio.run(aextract_and_save_all_info(
    #     rag_pipeline = rag_obj,
    #     nome_delle_aziende = nome_delle_aziende,
    #     llm_json = llm_for_extraction,
    #     save_dir = OUTPUT_SAVE_DIR,
    #     # optimized_queries = azienda_optmized_queries # TODO: REMOVE AFTER TESTING
    # ))

    logger.info("Completed Extraction. Outputs saved to %s", OUTPUT_SAVE_DIR)
    logger.info("Total time taken to run the script: %s", time.strftime("%H:%M:%S", time.gmtime(time.time()-t0)))




