from langchain_core.prompts import ChatPromptTemplate

from tqdm import tqdm


# python native
from pathlib import Path
from typing import Dict, Type, List, Tuple
import json
import textwrap
import datetime

# from other modules
from rag_info_extractor.info_schema.utils import formatted_word_to_number, load_classes_from_path, group_classes_by_module
from rag_pipeline import RAGPipeline #rag_information_extractor/scripts/rag_pipeline.py
from rag_info_extractor.llm_connector import OllamaLLM


## ---------------------------------------------- ******* ------------------------------------------
## ---------------------------------------------- HELPERS ------------------------------------------

# Define Extracter Prompt and LLM 
EXTRACTOR_SYSTEM_PROMPT = textwrap.dedent("""\
    SYSTEM:

    Sei un normalizzatore di RISPOSTE per un sistema RAG su statuti societari.
                                          
    Compila i campi del modello esclusivamente usando la RISPOSTA fornita qui sotto (non usare il contesto); mantieni il testo nei campi il più breve possibile.
    Regole:
    - Se la RISPOSTA è esattamente "Non ho trovato la risposta nei documenti forniti" oppure non copre un campo → metti stringa vuota "".
    - Sì/No: usa "Sì" o "No".
    - Elenchi: elementi separati da virgole, senza punto finale.
    - Numeri e date: riporta esattamente il testo così come compare nella RISPOSTA, senza convertirli né modificarli. Non convertire/normalizzare (non trasformare “duecento” in “200”)
    - Output: SOLO il JSON del modello richiesto; nessun testo extra; 

    HUMAN:
    
    DOMANDA:
    {question}
                            
    RISPOSTA:
    {answer}

    Compila il modello {sub_module} usando SOLO la RISPOSTA.                          
""")

# Define Post-Processing functions for extractions
POST_FUNCS = {
    "formatted_word_to_number": formatted_word_to_number, # type:ignore (from utility functions)
    }


## ---------------------------------------------- ******* ------------------------------------------
## ---------------------------------------------- MAIN  ------------------------------------------

class ExtractInfo:
    """Extract information from Schema"""

    def __init__(
        self,
        llm: OllamaLLM,
        extractor_graph: RAGPipeline,
        nome_azienda: str,
        sub_modules: List = []
    ) -> None:
        
        self.llm = llm
        self.sub_modules = sub_modules
        self.extractor_graph = extractor_graph
        self.nome_azienda = nome_azienda
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

        
        prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{question}", question).replace("{answer}", answer).replace("{sub_module}", sub_module.model_json_schema()['title'])
        
        result = self.llm.invoke(
            output_format = "structured",
            info_schema = sub_module,
            memory = prompt_content,
            num_predict = 512,
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
                val = data.get(field_name, "")
                data[field_name] = fn(val)

        return sub_module(**data)

    def extract_info(self):
        for m in self.sub_modules:
            logger.info(f"## \t Extracting info for moudle {m}: ")

            # Re-set time
            self.extractor_graph.reset_latency()

            logger.debug(f'Nome Azienda in : {self.nome_azienda.upper()}')### 
            q = m.question + f" Nome della società: {self.nome_azienda.upper()}" #  [{self.nome_azienda.upper()}]
            answer = self.extractor_graph.get_response(q)
            time_consumed = self.extractor_graph.latency

            name_sub_module: str = m.model_json_schema()['title']###
            
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


            logger.debug(f'Module: {name_sub_module}')###
            logger.debug(f"Question: \t {q}")###
            logger.debug(f"Answer: \t {answer}")###

            ## Save retrieved contexts and optimized query for testing
            self.optimized_query[name_sub_module] = self.extractor_graph.optimized_query###
            self.ori_contexts[name_sub_module] = self.extractor_graph.retrieved_docs_ids###
            self.ori_contexts_texts[name_sub_module] = self.extractor_graph.retrieved_docs_texts###
            try:
                self.re_ranked_contexts[name_sub_module] = self.extractor_graph.re_ranked_docs_ids###
                self.re_ranked_contexts_texts[name_sub_module] = self.extractor_graph.re_ranked_docs_texts###
            except AttributeError:
                self.re_ranked_contexts[name_sub_module] = {}#[]
                self.re_ranked_contexts_texts[name_sub_module] = {}###
            self.rag_qa[name_sub_module] = {"Q": q, "A": answer}###

            # print("Time taken:", time_consumed)
            time_consumed['overall'] = f"{float(time_consumed.get('overall', '0 s')[:-2]) + float(time_consumed.get('extract_sub_module', '0 s')[:-2])} s"###
            self.run_times[name_sub_module] = time_consumed###

            logger.debug(f'Time consumed on {name_sub_module}: {time_consumed}')###
            
    
    @property
    def output(self):
        return self.out


def extract_and_save_all_info(
    rag_pipeline: RAGPipeline,
    nome_delle_aziende: List[Tuple[str, str]],
    llm_json: OllamaLLM,
    save_dir: str
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
                sub_modules = info_group_modules
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
            logger.debug(f'{"-"*40} xxxxx {"-"*40}')

            # save info for all aziende
            info_extracted[azienda[0]] = info_per_azienda
            # save check point
            with open(f"{save_dir}/last_run.json", "w", encoding="utf-8") as f:
                json.dump(info_extracted, f, indent=4, ensure_ascii=False)

            
        # save info for all aziende
        info_extracted[azienda[0]] = info_per_azienda
        logger.info(f"Information extracted for the azienda: {azienda}.")

        # save checkpoint
        with open(f"{save_dir}/last_run.json", "w", encoding="utf-8") as f:
            json.dump(info_extracted, f, indent=4, ensure_ascii=False)
    
    # save final checkpoint
    with open(f"{save_dir}/last_run.json", "w", encoding="utf-8") as f:
        json.dump(info_extracted, f, indent=4, ensure_ascii=False)
    
    # write final result to output.json
    file_name = f"{save_dir}/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(info_extracted, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    
    from langchain_chroma import Chroma
    from langchain.storage import LocalFileStore
    from langchain_huggingface import HuggingFaceEmbeddings

    import os
    import yaml
    import time
    import argparse

    # logging relative
    import logging
    from rag_info_extractor.common_logging import configure_logging
    logger = logging.getLogger(__name__)


    t0 = time.time()

    # CONFIG FILE SETTINGS  (Load args form config file)
    cfg_path = Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    cfgs = configs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    RERANKER_MODEL = cfgs.get("RERANKER_MODEL")
    PRUNER_MODEL = cfgs.get("PRUNER_MODEL")
    LLM_MODEL = cfgs.get("LLM_MODEL")
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM")
    EXTRACTOR_LLM = cfgs.get("EXTRACTOR_LLM")

    USE_CUSTOM_CHUNKING = cfgs.get("USE_CUSTOM_CHUNKING")
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    MAX_EMBED_TOKENS = cfgs.get("MAX_EMBED_TOKENS")
    READ_MODE = cfgs.get("READ_MODE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    
    
    DATASET_DIR = os.path.join(BASE_DIR, "data", "documents", DATASET_TYPE) #f"../data/documents/{DATASET_TYPE}"
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) # f"../data/large_chunks_dbs/{DATASET_TYPE}/{CHUNKS_TYPE}"
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE) # f"../data/vector_dbs/{DATASET_TYPE}/{CHUNKS_TYPE}"
    OUTPUT_SAVE_DIR = os.path.join(BASE_DIR, 'outputs', DATASET_TYPE, CHUNKS_TYPE)

    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO, logfile=os.path.join(BASE_DIR, "info_extractor.log"))

    # Load Vector and Doc store
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                  encode_kwargs={"normalize_embeddings": True})
    vector_store = Chroma(embedding_function=embedding,
                        persist_directory=VECTOR_STORE_PATH,
                        collection_name="pdf_chunks")
    retriever = vector_store.as_retriever(search_type="similarity",
                                        search_kwargs={'k': 8})

    doc_store_page_content = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content") 
    doc_store_metadata = LocalFileStore(f"{DOC_STORE_LARGE_CHUNKS_PATH}/metadata")

    # Get all azienda names in vector/doc store
    nome_delle_aziende = set((vector_store.get()['metadatas'][i].get('azienda'), vector_store.get()['metadatas'][i].get('filename')) for i in range(len(vector_store.get()['ids']))) 
    nome_delle_aziende = sorted(nome_delle_aziende, key=lambda x: x[1])
    azienda_name_records = [x[0] for x in nome_delle_aziende]

    # Load the schemas of info to be etracted
    classes = load_classes_from_path(os.path.join(BASE_DIR, "src", "rag_info_extractor", "info_schema", "schemas"))

    # Load RAG pipeline
    logger.info("Initializing RAG Pipeline...") ###
    try:
        rag_obj = RAGPipeline(
            db_retriever = retriever,
            # pruner_model = PRUNER_MODEL,
            reranker_model = RERANKER_MODEL, 
            azienda_name_records = azienda_name_records,
            llm_model = LLM_MODEL,
            doc_store_path = DOC_STORE_LARGE_CHUNKS_PATH,
            pages_joining_str = PAGES_JOINING_STR
        ) 
        logger.info("Initialized RAG Pipeline.") ###
    except Exception as e:
        logger.exception(e) 
    

    # Define LLM for extractions
    llm_for_extraction = OllamaLLM(
        llm_model=EXTRACTOR_LLM,
        temperature=0
    )

    # Extract info
    extract_and_save_all_info(
        rag_pipeline = rag_obj,
        nome_delle_aziende = nome_delle_aziende,
        llm_json = llm_for_extraction,
        save_dir = OUTPUT_SAVE_DIR
    )

    logger.info(f"Completed Extraction. Outputs saved to {OUTPUT_SAVE_DIR}")
    logger.info(f'Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}')