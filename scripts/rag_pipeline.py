from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
# Detect if generated answer is in italiano for verifier node
from langdetect import DetectorFactory, detect
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from transformers import AutoModel

DetectorFactory.seed = 0 # probablistic algo.

import asyncio
# logging relative
import logging
import re
# Python native
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

import aiofiles
from rag_info_extractor.rag_pipeline.analyze_query import (Search,
                                                           aanalyze_query,
                                                           analyze_query)
from rag_info_extractor.rag_pipeline.generator import agenerate, generate
from rag_info_extractor.rag_pipeline.re_ranker import (
    across_encode_rerank, afaster_retrieve_and_rerank, cross_encode_rerank,
    faster_retrieve_and_rerank)
from rag_info_extractor.rag_pipeline.retrieve import aretrieve, retrieve
from rag_info_extractor.utils.apis_connector import (acall_pruner_service,
                                                     call_pruner_service)
# Import rag parts from modules
from rag_info_extractor.utils.llm_connector import OllamaLLM

# from rag_info_extractor.rag_pipeline.verify import verify

logger = logging.getLogger(__name__)




# TODO: 
# Aggiungere se query non menziona l'azienda, farlo per ogni azienda con citazione

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    rerank_debug: Dict
    answer: str



class RAGPipeline:

    def __init__(
        self,
        db_retriever: VectorStoreRetriever,
        azienda_name_records: List[str],
        llm_model: str,
        doc_store_path: Optional[str],
        pages_joining_str: Optional[str],
        run_async: bool = False,
        use_google_api: bool = False
        
    ):
        """
        Args:
            db_retriever (VectorStoreRetriever): retriver function for the vector database.
            azienda_name_records (List[str]): names of all azienda/società for which the record exist in db
            doc_store_path (Optional[str]): path of the folder containing parent/large chunks
        """

        # Initialize/Load the vector db and doc_store and llm
        self.retriever = db_retriever
        self.llm = OllamaLLM(llm_model = llm_model)

        # Retrieve full/Parent chunks if doc_store_path passed
        if doc_store_path:
            self.doc_store_path = doc_store_path
        else:
            self.doc_store_page_content = None
            self.doc_store_metadata = None

        # Set azienda names in db
        self.azienda_name_records = azienda_name_records

        # Other inits
        self.pages_joining_str = pages_joining_str
        self.use_google_api = use_google_api
        
        # Initialize the langgraph
        if run_async:
            graph = (
                StateGraph(State)
                .add_node("analyze_query", self.arun_analyze_query)           
                .add_node("retrieve", self.arun_retrieve)
                .add_node("cross_encode_rerank", self.arun_cross_encode_rerank)
                .add_node("pruning", self.arun_pruning)
                .add_node("generate", self.arun_generate)
                .add_node("faster_retrieve_and_rerank", self.arun_faster_retrieve_and_rerank)
                # .add_node("verify", self.arun_verify)    
            )
        else:
            graph = (
                StateGraph(State)
                .add_node("analyze_query", self.run_analyze_query)           
                .add_node("retrieve", self.run_retrieve)
                .add_node("cross_encode_rerank", self.run_cross_encode_rerank)
                .add_node("pruning", self.run_pruning)
                .add_node("generate", self.run_generate)
                .add_node("faster_retrieve_and_rerank", self.run_faster_retrieve_and_rerank)
                # .add_node("verify", self.run_verify)    
            )

        # add edges to the graph
        graph.set_entry_point("analyze_query")

        graph.add_edge("analyze_query", "retrieve")
        # graph.add_edge("analyze_query", "faster_retrieve_and_rerank")
        
        graph.add_edge("retrieve", "cross_encode_rerank")
        # graph.add_edge("retrieve", "pruning")
        # graph.add_edge("cross_encode_rerank", "pruning")

        graph.add_edge("cross_encode_rerank", "generate")
        # graph.add_edge("pruning", "generate")
        # graph.add_edge("retrieve", "generate")
        # graph.add_edge("faster_retrieve_and_rerank", "generate")

        # graph.add_edge("generate", "verify")
        
        graph.add_edge("generate", END)
        # graph.add_edge("verify", END)
    
        self.graph = graph.compile()

        # For calculation of latency
        self.latency = {k: "0.00 s" for k in
                        ["analyze_query","retrieve","pruning","generate", "re_ranking", "faster_retrieve_and_rerank", "overall"]}
        # For storing responses for testing
        self.retrieved_docs_ids: Dict[str, List[int]] = {} 
        self.re_ranked_docs_ids: Dict[str, List[int]] = {}
        self.retrieved_docs_texts: Dict[str, List[str]] = {}
        self.re_ranked_docs_texts: Dict[str, List[str]] = {}
        self.optimized_query: Dict[str, str] = {}

    def reset_latency(self):
        for k in self.latency:
            self.latency[k] = "0.00 s"


    
    # =========== NODES ====================

    # --------- ANALYZE QUERY ----------
    def run_analyze_query(self, state: State, nome_azienda: str=""):
        """
        Re-phrase query to improve it
        
        nome_azienda (str): name of the società for which the information needs to be extracted 
        """
        
        t1 = time.time()
        
        logger.debug(f"\nInputs to analyze_query:\nquestion: {state['question']}\nazienda_name_records: {self.azienda_name_records}\nuse_google_api: {self.use_google_api}\n")
        response: Search = analyze_query(
            question = state['question'],
            llm = self.llm,
            nome_azienda = "",
            azienda_name_records = self.azienda_name_records,
            use_google_api = self.use_google_api
        )
        # ==================== FOR QUICK TESTING =========================================
        # question, optimized_query, azienda_name= re.findall(r"(.*) QUERY: (.*) Nome della società: (.*)", state['question'])[0] # TODO: REMOVE AFTER TESTING
        # state['question'] = question
        # response = Search(query=optimized_query, azienda=azienda_name) # TODO: REMOVE AFTER TESTING
        # =========================   XXX   =========================================
        
        self.latency['analyze_query'] = f"{time.time() - t1:.3f} s"
        # save query for testing
        self.optimized_query = {"query": response.query, "azienda": response.azienda}###

        logger.debug(f"\nOutput of analyze_query:\n'query': {response}\n\n")
        return {"query": response}
    
    # Async version
    async def arun_analyze_query(self, state: State, nome_azienda: str=""):
        """
        Re-phrase query to improve it
        
        nome_azienda (str): name of the società for which the information needs to be extracted 
        """
        
        t1 = time.time()
        
        logger.debug(f"\nInputs to (async) analyze_query:\nquestion: {state['question']}\nazienda_name_records: {self.azienda_name_records}\nuse_google_api: {self.use_google_api}\n")
        response: Search = await aanalyze_query(
            question = state['question'],
            llm = self.llm,
            nome_azienda = "",
            azienda_name_records = self.azienda_name_records,
            use_google_api = self.use_google_api
        )
        
        self.latency['analyze_query'] = f"{time.time() - t1:.3f} s"
        self.optimized_query = {"query": response.query, "azienda": response.azienda}###

        logger.debug(f"\nOutput of (async) analyze_query:\n'query': {response}\n\n")
        return {"query": response}

    # --------- RETRIEVER ----------
    def run_retrieve(self, state: State):
        
        t1 = time.time()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(f"\nInputs to retrieve:\nquery: {query}\nazienda: {azienda}\n")
        output_retrieve = retrieve(
            retriever = self.retriever,
            query = query,
            doc_store_large_chunks_path = self.doc_store_path,
            azienda = azienda,
            pages_joining_str = self.pages_joining_str,
            retrieve_parents = False,
            save_full_chunks = False
        )

        retrieved_docs = output_retrieve.get("context", [])
        self.retrieved_docs_ids = output_retrieve.get("retrieved_docs_ids", {})
        self.retrieved_docs_texts = output_retrieve.get("retrieved_docs_texts", {})

        self.latency['retrieve'] = f"{time.time() - t1:.3f} s"
        
        logger.debug(f"\nOutput of retrieve:\n'context': {retrieved_docs}\n\n")
        return {"context": retrieved_docs} #retrieved_docs_parent

    # Async version
    async def arun_retrieve(self, state: State):
        
        t1 = time.time()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(f"\nInputs to (async) retrieve:\nquery: {query}\nazienda: {azienda}\n")
        output_retrieve = await aretrieve(
            retriever = self.retriever,
            query = query,
            doc_store_large_chunks_path = self.doc_store_path,
            azienda = azienda,
            pages_joining_str = self.pages_joining_str,
            retrieve_parents = False, # TODO: DEFAULT=False
            save_full_chunks = False
        )

        retrieved_docs = output_retrieve.get("context", [])
        self.retrieved_docs_ids = output_retrieve.get("retrieved_docs_ids", {})
        self.retrieved_docs_texts = output_retrieve.get("retrieved_docs_texts", {})

        self.latency['retrieve'] = f"{time.time() - t1:.3f} s"
        
        logger.debug(f"\nOutput of (async) retrieve:\n'context': {retrieved_docs}\n\n")
        return {"context": retrieved_docs} #retrieved_docs_parent

    # --------- PRUNER ----------
    def run_pruning(self, state: State):
        # Execute pruning
        ori_context = [c.page_content for c in state['context']]
        t1 = time.time()

        logger.debug(f"\nInputs to pruning:\nquery: {state['question']}\ndocuments: {ori_context}\n")
        pruned_docs = call_pruner_service(
            query = state['question'],
            documents = ori_context
        )

        self.latency['pruning'] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of pruning:\n'context': {pruned_docs}\n\n")
        return {"context": [Document(page_content=c) for c in pruned_docs]}

    # Async version
    async def arun_pruning(self, state: State):
        # Execute pruning  
        ori_context = [c.page_content for c in state['context']]
        t1 = time.time()

        logger.debug(f"\nInputs to (async) pruning:\nquery: {state['question']}\ndocuments: {ori_context}\n")
        pruned_docs = await acall_pruner_service(
            query = state['question'],
            documents = ori_context
        )

        self.latency['pruning'] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) pruning:\n'context': {pruned_docs}\n\n")
        return {"context": [Document(page_content=c) for c in pruned_docs]}

    # --------- RE-RANKER ----------
    def run_cross_encode_rerank(self, state: State) -> Dict[str, List[Document]]:
        """Re-ranks the retrieved docs"""
    
        t1 = time.time()

        logger.debug(f"\nInputs to cross_encode_rerank:\ncontexts: {state['context']}\nquestion: {state['question']}\n")
        re_ranker_output = cross_encode_rerank(
            contexts = state["context"],
            question = state["question"],
            doc_store_large_chunks_path = self.doc_store_path,
            k_min = 2,
            k_max = 5,
            rel_thresh = 0.4,
            max_promoted_parents = 3,
            use_parent_heuristics = False,
            save_full_chunks = False
        )

        re_ranked_docs = re_ranker_output.get("context", [])
        self.re_ranked_docs_ids = re_ranker_output.get("re_ranked_docs_ids", {})
        self.re_ranked_docs_texts = re_ranker_output.get("re_ranked_docs_texts", {})
        re_ranker_debug_info = re_ranker_output.get("re_rank_debug", {})

        # save debug info
        state["rerank_debug"] = re_ranker_debug_info

        self.latency['re_ranking'] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of cross_encode_rerank:\n'context': {re_ranked_docs}\n\n")
        return {"context": re_ranked_docs} ### re_ranked_docs_parent

    # Async version
    async def arun_cross_encode_rerank(self, state: State) -> Dict[str, List[Document]]:
        """Async implementation of run_cross_encode_rerank"""
    
        t1 = time.time()
        
        logger.debug(f"\nInputs to (async) cross_encode_rerank:\ncontexts: {state['context']}\nquestion: {state['question']}\n")
        re_ranker_output = await across_encode_rerank(
            contexts = state["context"],
            question = state["question"],
            doc_store_large_chunks_path = self.doc_store_path,
            k_min = 2,
            k_max = 5,
            rel_thresh = 0.4,
            max_promoted_parents = 3,
            use_parent_heuristics = False,
            save_full_chunks = False
        )

        re_ranked_docs = re_ranker_output.get("context", [])
        self.re_ranked_docs_ids = re_ranker_output.get("re_ranked_docs_ids", {})
        self.re_ranked_docs_texts = re_ranker_output.get("re_ranked_docs_texts", {})
        re_ranker_debug_info = re_ranker_output.get("re_rank_debug", {})

        # save debug info
        state["rerank_debug"] = re_ranker_debug_info

        self.latency['re_ranking'] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) cross_encode_rerank:\n'context': {re_ranked_docs}\n\n")
        return {"context": re_ranked_docs} ### re_ranked_docs_parent

    # --------- FAST RETRIEVER + RE-RANKER ----------
    def run_faster_retrieve_and_rerank(self, state: State):
        """Retrieve + Re-rank + Compress context"""
        t1 = time.time()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(f"\nInputs to faster_retrieve_and_rerank:\nquery: {query}\nazienda: {azienda}\n")
        output = faster_retrieve_and_rerank(
            query = query,
            retriever = self.retriever,
            azienda = azienda,
            top_n = 6,
            pages_joining_str = self.pages_joining_str,
            save_full_chunks = True # default=False
        )

        docs = output.get("context", [])
        self.re_ranked_docs_ids = output.get("docs_ids", {})
        self.re_ranked_docs_texts = output.get("docs_texts", {})
    

        self.latency['faster_retrieve_and_rerank'] = f"{time.time() - t1:.3f} s" 
        
        logger.debug(f"\nOutput of faster_retrieve_and_rerank:\n'context': {docs}\n\n")
        return {"context": docs}

    # Async version
    async def arun_faster_retrieve_and_rerank(self, state: State):
        """Async version of run_faster_retrieve_and_rerank"""
        t1 = time.time()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(f"\nInputs to (async) faster_retrieve_and_rerank:\nquery: {query}\nazienda: {azienda}\n")
        output = await afaster_retrieve_and_rerank(
            query = query,
            retriever = self.retriever,
            azienda = azienda,
            top_n = 6,
            pages_joining_str = self.pages_joining_str,
            save_full_chunks = True # default=False
        )

        docs = output.get("context", [])
        self.re_ranked_docs_ids = output.get("docs_ids", {})
        self.re_ranked_docs_texts = output.get("docs_texts", {})
    

        self.latency['faster_retrieve_and_rerank'] = f"{time.time() - t1:.3f} s" 
        
        logger.debug(f"\nOutput of (async) faster_retrieve_and_rerank:\n'context': {docs}\n\n")
        return {"context": docs}

    # --------- GENERATOR ----------
    def run_generate(self, state: State):
        
        t1 = time.time()

        logger.debug(f"\nInputs to generate:\nquestion: {state['question']}\ncontexts:\n {state['context']}\n")
        answer = generate(
            question = state["question"],
            contexts = state["context"],
            llm = self.llm,
            contexts_sep = "||",
            use_google_api = self.use_google_api
        )
        

        # detect if answer in italiano, if not regenerate (1 try max to avoid constant loop)
        if detect(answer) != "it":
            print("Regenerating answer....")###
            answer = generate(
                question = state["question"],
                contexts = state["context"],
                llm = self.llm,
                contexts_sep = "||",
                additional_prompt = "Rispondere sempre in Italiano.",
                use_google_api = self.use_google_api
            )
        
        self.latency['generate'] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of generate:\n'answer': {answer}\n\n")
        return {"answer": answer}

    # Async version
    async def arun_generate(self, state: State):
        """Async version run_generate"""
        t1 = time.time()

        logger.debug(f"\nInputs to (async) generate:\nquestion: {state['question']}\ncontexts:\n {state['context']}\n")
        answer = await agenerate(
            question = state["question"],
            contexts = state["context"],
            llm = self.llm,
            contexts_sep = "||",
            use_google_api = self.use_google_api
        )

        # detect if answer in italiano, if not regenerate (1 try max to avoid constant loop)
        if detect(answer) != "it":
            print("Regenerating answer....")###
            answer = await agenerate(
                question = state["question"],
                contexts = state["context"],
                llm = self.llm,
                contexts_sep = "||",
                additional_prompt = "Rispondere sempre in Italiano.",
                use_google_api = self.use_google_api
            )
        
        self.latency['generate'] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) generate:\n'answer': {answer}\n\n")
        return {"answer": answer}

    # ---------- VERIFIER --------------
    def run_verify(self, state: State):
        # TODO: Implement verifying logic
        # FALLBACK_IT = "Non ho trovato la risposta nei documenti forniti"

        # draft_ans = state.get("answer", "") or ""
        # contexts = state.get("contexts", []) or []
        # verdict = self.verifier.verify(draft_ans, contexts)
        # Single check → return immediately with either original or fallback
        # final_ans = draft_ans if verdict["all_supported"] else FALLBACK_IT
        
        # return {"answer": final_ans}
        pass

    # =========== XXX ====================
    
    # other methods
    def get_response(self, query: str, nome_azienda: str = "") -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
    
        try:
            t1 = time.time()
            response = self.graph.invoke({"question": query}) # type: ignore 
            self.latency['overall'] = f"{time.time() - t1:.3f} s"
            return response['answer']  
        except Exception as e:
            print(f"Exception: {e}")
            return ""

    # Async version
    async def aget_response(self, query: str, nome_azienda: str = "") -> str:
        """Async version of get_response"""
        try:
            t1 = time.time()
            response = await self.graph.ainvoke({"question": query}) # type: ignore 
            self.latency['overall'] = f"{time.time() - t1:.3f} s"
            return response['answer']  
        except Exception as e:
            print(f"Exception: {e}")
            logger.exception(f"Exception: {e}")
            return ""


    def save_DAG_diagram(self, directory: str=""):
        if directory:
            filename = f"{directory}/rag_pipeline.png"
        else:
            filename = "rag_pipeline.png"
        try:
            dag_img = self.graph.get_graph().draw_mermaid_png(max_retries=5)
        except:
            logger.exception("Error! DAG for the RAG Pipeline not generated")
        else:
            with open(filename, "wb") as png:
                png.write(dag_img)
    
    # Async version
    async def asave_DAG_diagram(self, directory: str=""):
        if directory:
            filename = f"{directory}/rag_pipeline.png"
        else:
            filename = "rag_pipeline.png"
        try:
            dag_img = self.graph.get_graph().draw_mermaid_png(max_retries=5)
        except:
            logger.exception("Error! DAG for the RAG Pipeline not generated")
        else:
            async with aiofiles.open(filename, "wb") as png:
                await png.write(dag_img)



if __name__ == "__main__":

    import argparse
    import datetime
    # logging relative
    import logging
    import os
    import time
    from pathlib import Path

    import yaml
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from rag_info_extractor.utils.common_logging import configure_logging
    from rag_info_extractor.utils.embedder import HFEmbedder
    from rag_info_extractor.utils.load_config import cfgs
    logger = logging.getLogger(__name__)

    t0 = time.time()

    # CONFIG FILE SETTINGS:
    cfgs = cfgs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    LLM_MODEL = cfgs.get("LLM_MODEL") 
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    CHUNKS_TYPE = cfgs.get("CHUNKS_TYPE")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    RERANKER_MODEL = cfgs.get("RERANKER_MODEL")
    PRUNER_MODEL = cfgs.get("PRUNER_MODEL")
    RUN_ASYNC = cfgs.get("RUN_ASYNC", False)
    USE_GOOGLE_API = cfgs.get("USE_GOOGLE_API", False)

    
    DOC_STORE_LARGE_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "large_chunks_dbs", DATASET_TYPE, CHUNKS_TYPE) 
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_dbs", DATASET_TYPE, CHUNKS_TYPE)
    assert os.path.exists(DOC_STORE_LARGE_CHUNKS_PATH), "DOC_STORE_LARGE_CHUNKS_PATH not found"
    assert os.path.exists(VECTOR_STORE_PATH), "VECTOR_STORE_PATH not found"

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") 
    args = parser.parse_args()
    
    # Configure logging settings
    RUN_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    LOGDIR = os.path.join(BASE_DIR, "logs", "rag_pipeline_py")
    os.makedirs(LOGDIR, exist_ok=True)
    configure_logging(
        default_level=logging.DEBUG if args.verbose else logging.INFO,
        logfile=os.path.join(LOGDIR, f"{RUN_TIME}.log")
    )
    

    logger.info(f'Logging for {"-"*30} rag_information_extractor/scripts/rag_pipeline.py') ###
    logger.info(f'LLM model used: {LLM_MODEL}')
    
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
    azienda_name_records = [x[0] for x in nome_delle_aziende]

    logger.info("Loaded Vector + Doc Store.") ###

    # Initialize RAG pipeline
    logger.info("Initializing RAG Pipeline...") ###
    try:
        rag_obj = RAGPipeline(
            db_retriever = retriever,
            azienda_name_records = azienda_name_records,
            llm_model = LLM_MODEL,
            doc_store_path = DOC_STORE_LARGE_CHUNKS_PATH,
            pages_joining_str = PAGES_JOINING_STR,
            run_async = RUN_ASYNC,
            use_google_api = USE_GOOGLE_API
        ) 
        logger.info("Initialized RAG Pipeline.") ###
    except:
        logger.exception("message")
    else:
        # save the DAG flow for RAG nodes/Pipeline
        rag_obj.save_DAG_diagram("")
        logger.info(f"The DAG for RAG Pipeline saved to {""}") 


    # Run RAG pipeline
    USER_QUERY = "Agli amministratori spetta il rimborso delle spese? Informazione richiesto per la società: 2KIND SRL"  # Query and Aziende (EXAMPLE)
    # USER_QUERY = "I soci possono assegnare un compenso agli amministratori? In che misura? QUERY: compenso amministratori limiti massimi e criteri di determinazione indennità Nome della società: 2kind srl"
    logger.info("Running query...") ###
    if RUN_ASYNC:
        ai_response = asyncio.run(rag_obj.aget_response(query=USER_QUERY))
    else:
        ai_response = rag_obj.get_response(query=USER_QUERY)

    
    # Save outputs to output_temp.txt
    def write_outputs_to_file():
        # Obtain retrieved chunks texts
        retrieved_docs_ids = rag_obj.retrieved_docs_ids
        retrieved_vs_chunk_ids = [i for i, m in enumerate(vector_store.get().get("metadatas", [])) if m.get("chunk_id") in retrieved_docs_ids.get("children", [])]
        retrieved_vs_chunks = [c for i, c in enumerate(vector_store.get().get("documents", [])) if i in retrieved_vs_chunk_ids] # vector_store chunks
        
        retrieved_parents_keys = retrieved_docs_ids.get("parents", [])
        retrieved_ds_chunks = ["" for i in range(len(retrieved_parents_keys))]
        for i, id in enumerate(retrieved_parents_keys):
            with open(f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content/{id}", encoding="utf-8") as f:
                retrieved_ds_chunks[i] = f.read()

        # Obtain re_ranked_chunks
        re_ranked_docs_ids = rag_obj.re_ranked_docs_ids
        re_ranked_vs_chunk_ids = [i for i, m in enumerate(vector_store.get().get("metadatas", [])) if m.get("chunk_id") in re_ranked_docs_ids.get("children", [])]
        re_ranked_vs_chunks = [c for i, c in enumerate(vector_store.get().get("documents", [])) if i in re_ranked_vs_chunk_ids] # vector_store chunks
        
        re_ranked_parents_keys = re_ranked_docs_ids.get("parents", [])
        re_ranked_ds_chunks = ["" for i in range(len(re_ranked_parents_keys))]
        for i, id in enumerate(re_ranked_parents_keys):
            with open(f"{DOC_STORE_LARGE_CHUNKS_PATH}/page_content/{id}", encoding="utf-8") as f:
                re_ranked_ds_chunks[i] = f.read()

        # Store contexts/query in output_temp.txt
        with open("output_temp", "w", encoding="utf-8") as f:
            f.write(f"## OUTPUT FOR: rag_pipeline.py \n{RUN_TIME}\n")
            f.write(f"Async: {RUN_ASYNC}\n\n")
            f.write(f"USER QUERY: {USER_QUERY}\n\n")
            f.write(f"Optimied Query: {rag_obj.optimized_query}\n\n")
            f.write(f"LLM ANSWER: {ai_response}\n\n")
            
            f.write(f"\n{"x"*100}\n")

            # Retrieved documents
            f.write("\nRETRIEVER DOCs...\n\n")
            f.write(f"Retrieved doc ids: {retrieved_docs_ids}\n\n")
            f.write(f"VECTOR STORE chunks: \n")
            for i, c in enumerate(retrieved_vs_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")
            f.write(f"{"-"*100}\n{"-"*100}\n")
            f.write(f"DOC STORE chunks: \n")
            for i, c in enumerate(retrieved_ds_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")


            f.write(f"\n{"x"*100}\n")

            # Re-Ranked documents
            f.write("\nRE-RANKED DOCs...\n\n")
            f.write(f"Re-Ranked doc ids: {re_ranked_docs_ids}\n\n")
            f.write(f"VECTOR STORE chunks: \n")
            for i, c in enumerate(re_ranked_vs_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")
            f.write(f"{"-"*100}\n{"-"*100}\n")
            f.write(f"DOC STORE chunks: \n")
            for i, c in enumerate(re_ranked_ds_chunks):
                f.write(f"\n{"-"*50} CHUNK {i} {"-"*50}\n")
                f.write(f"{c}\n\n")
    
    logger.info("Saving outputs...")
    write_outputs_to_file()
    
    logger.info(f'Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}')