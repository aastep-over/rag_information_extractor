import pytest
import os
import json
import asyncio
from typing import Dict, List, Optional, TypedDict, cast, Any
import logging
import time
import random
from pathlib import Path

from rag_info_extractor.utils.llm_connector import OllamaLLM
from rag_info_extractor.rag_pipeline import RAGPipeline
from langgraph.graph import END, START, StateGraph
from rag_info_extractor.extract_info import (
    extract_with_GOOGLE_API, aextract_with_GOOGLE_API, EXTRACTOR_SYSTEM_PROMPT, POST_FUNCS,
    extract_and_save_all_info,
    aextract_and_save_all_info,
    load_info_to_extract_classes
)
import rag_info_extractor.extract_info as extract_info_module
from rag_info_extractor.rag_pipeline_components.analyze_query import Search
from pydantic import BaseModel, Field
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================= Test Google APIs =====================================
def test_extract_with_GOOGLE_API():

    # Define class with Pydantic
    class InfoSchema(BaseModel):
        capital: str = Field(..., description="Il capitale del paese.")
    
    # Define prompt content
    prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{sub_module_description}", json.dumps(InfoSchema.model_json_schema()['properties'], indent=2, ensure_ascii=False))
    answer="Italia è un paese mediterrano e ha Roma come capitale."
    prompt_content = prompt_content.replace("{answer}", answer)
    

    response = extract_with_GOOGLE_API(prompt_content, InfoSchema) # type: ignore
    assert response["capital"] == "Roma"

async def test_extract_with_GOOGLE_API_async():
    # Define class with Pydantic
    class InfoSchema(BaseModel):
        capital: str = Field(..., description="Il capitale del paese.")
    
    # Define prompt content
    prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("{sub_module_description}", json.dumps(InfoSchema.model_json_schema()['properties'], indent=2, ensure_ascii=False))
    answer="Italia è un paese mediterrano e ha Roma come capitale."
    prompt_content = prompt_content.replace("{answer}", answer)

    response = await aextract_with_GOOGLE_API(prompt_content, InfoSchema) # type: ignore
    assert response["capital"] == "Roma"


# =============================== PRE-DEFINED Information for Testing ========================================
def pre_def_info_for_test():
    with open(SCRIPT_DIR / "test_info_pred.json", encoding="utf-8") as f:
        test_info_json = json.load(f) # contains only 1 azienda data
        for k, v in test_info_json.items():
            nome_azienda, test_info = k, v
    # 1. BILANCI_E_UTILI 
    state_CapitaleSociale = State(
        question=test_info['BILANCI_E_UTILI']['rag_qa']['CapitaleSociale']['Q'],
        query=Search.model_validate(test_info['BILANCI_E_UTILI']['optimized_query']['CapitaleSociale']),
        context=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['CapitaleSociale']['children'],
        retrieved_docs_ids=test_info['BILANCI_E_UTILI']['retrieved_docs']['CapitaleSociale']['children'],
        retrieved_docs_texts=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['CapitaleSociale']['children'],
        re_ranked_context=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['CapitaleSociale']['parents'],
        re_ranked_docs_ids=test_info['BILANCI_E_UTILI']['re_ranked_docs']['CapitaleSociale']['parents'],
        re_ranked_docs_texts=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['CapitaleSociale']['parents'],
        rerank_debug={},
        answer=test_info['BILANCI_E_UTILI']['rag_qa']['CapitaleSociale']['A']
    )

    state_DataChiusuraEsercizio = State(
        question=test_info['BILANCI_E_UTILI']['rag_qa']['DataChiusuraEsercizio']['Q'],
        query=Search.model_validate(test_info['BILANCI_E_UTILI']['optimized_query']['DataChiusuraEsercizio']),
        context=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['DataChiusuraEsercizio']['children'],
        retrieved_docs_ids=test_info['BILANCI_E_UTILI']['retrieved_docs']['DataChiusuraEsercizio']['children'],
        retrieved_docs_texts=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['DataChiusuraEsercizio']['children'],
        re_ranked_context=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['DataChiusuraEsercizio']['parents'],
        re_ranked_docs_ids=test_info['BILANCI_E_UTILI']['re_ranked_docs']['DataChiusuraEsercizio']['parents'],
        re_ranked_docs_texts=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['DataChiusuraEsercizio']['parents'],
        rerank_debug={},
        answer=test_info['BILANCI_E_UTILI']['rag_qa']['DataChiusuraEsercizio']['A']
    )

    state_PercentualeRiservaLegale = State(
        question=test_info['BILANCI_E_UTILI']['rag_qa']['PercentualeRiservaLegale']['Q'],
        query=Search.model_validate(test_info['BILANCI_E_UTILI']['optimized_query']['PercentualeRiservaLegale']),
        context=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['PercentualeRiservaLegale']['children'],
        retrieved_docs_ids=test_info['BILANCI_E_UTILI']['retrieved_docs']['PercentualeRiservaLegale']['children'],
        retrieved_docs_texts=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['PercentualeRiservaLegale']['children'],
        re_ranked_context=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['PercentualeRiservaLegale']['parents'],
        re_ranked_docs_ids=test_info['BILANCI_E_UTILI']['re_ranked_docs']['PercentualeRiservaLegale']['parents'],
        re_ranked_docs_texts=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['PercentualeRiservaLegale']['parents'],
        rerank_debug={},
        answer=test_info['BILANCI_E_UTILI']['rag_qa']['PercentualeRiservaLegale']['A']
    )

    state_TermineApprovazioneBilancio = State(
        question=test_info['BILANCI_E_UTILI']['rag_qa']['TermineApprovazioneBilancio']['Q'],
        query=Search.model_validate(test_info['BILANCI_E_UTILI']['optimized_query']['TermineApprovazioneBilancio']),
        context=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['TermineApprovazioneBilancio']['children'],
        retrieved_docs_ids=test_info['BILANCI_E_UTILI']['retrieved_docs']['TermineApprovazioneBilancio']['children'],
        retrieved_docs_texts=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['TermineApprovazioneBilancio']['children'],
        re_ranked_context=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['TermineApprovazioneBilancio']['parents'],
        re_ranked_docs_ids=test_info['BILANCI_E_UTILI']['re_ranked_docs']['TermineApprovazioneBilancio']['parents'],
        re_ranked_docs_texts=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['TermineApprovazioneBilancio']['parents'],
        rerank_debug={},
        answer=test_info['BILANCI_E_UTILI']['rag_qa']['TermineApprovazioneBilancio']['A']
    )

    state_UtiliResidui = State(
        question=test_info['BILANCI_E_UTILI']['rag_qa']['UtiliResidui']['Q'],
        query=Search.model_validate(test_info['BILANCI_E_UTILI']['optimized_query']['UtiliResidui']),
        context=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['UtiliResidui']['children'],
        retrieved_docs_ids=test_info['BILANCI_E_UTILI']['retrieved_docs']['UtiliResidui']['children'],
        retrieved_docs_texts=test_info['BILANCI_E_UTILI']['retrieved_docs_texts']['UtiliResidui']['children'],
        re_ranked_context=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['UtiliResidui']['parents'],
        re_ranked_docs_ids=test_info['BILANCI_E_UTILI']['re_ranked_docs']['UtiliResidui']['parents'],
        re_ranked_docs_texts=test_info['BILANCI_E_UTILI']['re_ranked_docs_texts']['UtiliResidui']['parents'],
        rerank_debug={},
        answer=test_info['BILANCI_E_UTILI']['rag_qa']['UtiliResidui']['A']
    )

    # 2. COMPENSO_DEGLI_AMMINISTRATORI
    state_IndennitaAnnuale = State(
        question=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['rag_qa']['IndennitaAnnuale']['Q'],
        query=Search.model_validate(test_info['COMPENSO_DEGLI_AMMINISTRATORI']['optimized_query']['IndennitaAnnuale']),
        context=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs_texts']['IndennitaAnnuale']['children'],
        retrieved_docs_ids=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs']['IndennitaAnnuale']['children'],
        retrieved_docs_texts=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs_texts']['IndennitaAnnuale']['children'],
        re_ranked_context=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs_texts']['IndennitaAnnuale']['parents'],
        re_ranked_docs_ids=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs']['IndennitaAnnuale']['parents'],
        re_ranked_docs_texts=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs_texts']['IndennitaAnnuale']['parents'],
        rerank_debug={},
        answer=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['rag_qa']['IndennitaAnnuale']['A']
    )

    state_IndennitaCessazione = State(
        question=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['rag_qa']['IndennitaCessazione']['Q'],
        query=Search.model_validate(test_info['COMPENSO_DEGLI_AMMINISTRATORI']['optimized_query']['IndennitaCessazione']),
        context=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs_texts']['IndennitaCessazione']['children'],
        retrieved_docs_ids=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs']['IndennitaCessazione']['children'],
        retrieved_docs_texts=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs_texts']['IndennitaCessazione']['children'],
        re_ranked_context=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs_texts']['IndennitaCessazione']['parents'],
        re_ranked_docs_ids=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs']['IndennitaCessazione']['parents'],
        re_ranked_docs_texts=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs_texts']['IndennitaCessazione']['parents'],
        rerank_debug={},
        answer=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['rag_qa']['IndennitaCessazione']['A']
    )

    state_Rimborso = State(
        question=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['rag_qa']['Rimborso']['Q'],
        query=Search.model_validate(test_info['COMPENSO_DEGLI_AMMINISTRATORI']['optimized_query']['Rimborso']),
        context=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs_texts']['Rimborso']['children'],
        retrieved_docs_ids=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs']['Rimborso']['children'],
        retrieved_docs_texts=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['retrieved_docs_texts']['Rimborso']['children'],
        re_ranked_context=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs_texts']['Rimborso']['parents'],
        re_ranked_docs_ids=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs']['Rimborso']['parents'],
        re_ranked_docs_texts=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['re_ranked_docs_texts']['Rimborso']['parents'],
        rerank_debug={},
        answer=test_info['COMPENSO_DEGLI_AMMINISTRATORI']['rag_qa']['Rimborso']['A']
    )

    # 3. INFO_GENERALI
    state_Durata = State(
        question=test_info['INFO_GENERALI']['rag_qa']['Durata']['Q'],
        query=Search.model_validate(test_info['INFO_GENERALI']['optimized_query']['Durata']),
        context=test_info['INFO_GENERALI']['retrieved_docs_texts']['Durata']['children'],
        retrieved_docs_ids=test_info['INFO_GENERALI']['retrieved_docs']['Durata']['children'],
        retrieved_docs_texts=test_info['INFO_GENERALI']['retrieved_docs_texts']['Durata']['children'],
        re_ranked_context=test_info['INFO_GENERALI']['re_ranked_docs_texts']['Durata']['parents'],
        re_ranked_docs_ids=test_info['INFO_GENERALI']['re_ranked_docs']['Durata']['parents'],
        re_ranked_docs_texts=test_info['INFO_GENERALI']['re_ranked_docs_texts']['Durata']['parents'],
        rerank_debug={},
        answer=test_info['INFO_GENERALI']['rag_qa']['Durata']['A']
    )

    # Dictionary to map state info to state_name
    return {
        "CapitaleSociale": state_CapitaleSociale,
        "DataChiusuraEsercizio": state_DataChiusuraEsercizio,
        "PercentualeRiservaLegale": state_PercentualeRiservaLegale,
        "TermineApprovazioneBilancio": state_TermineApprovazioneBilancio,
        "UtiliResidui": state_UtiliResidui,
        "IndennitaAnnuale": state_IndennitaAnnuale,
        "IndennitaCessazione": state_IndennitaCessazione,
        "Rimborso": state_Rimborso,
        "Durata": state_Durata
    }


# ================================ Test ExtractInfo Class ================================
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    retrieved_docs_ids: List[int]
    retrieved_docs_texts: List[str]
    re_ranked_context: List[Document]
    re_ranked_docs_ids: List[int]
    re_ranked_docs_texts: List[str]
    rerank_debug: Dict
    answer: str

class DummyRAGPipeline(RAGPipeline):
    # TODO: Have to reimplement it using args and methods from RAGPipeline

    ## TODO: Methods to define
    # .reset_latency()
    # .get_response(q), aget_response

    def __init__(
        self,
        azienda_name_records: List[str],
        run_async: bool = False,
        use_google_api: bool = False,
    ): 
        self.use_google_api = use_google_api
        self.azienda_name_records = azienda_name_records

        # Initialize the langgraph
        if run_async:
            graph = (
                StateGraph(State)
                .add_node("analyze_query", self.arun_analyze_query)
                .add_node("retrieve", self.arun_retrieve)
                .add_node("cross_encode_rerank", self.arun_cross_encode_rerank)
                .add_node("pruning", self.arun_pruning)
                .add_node("generate", self.arun_generate)
                .add_node(
                    "faster_retrieve_and_rerank", self.arun_faster_retrieve_and_rerank
                )
            )
        else:
            graph = (
                StateGraph(State)
                .add_node("analyze_query", self.run_analyze_query)
                .add_node("retrieve", self.run_retrieve)
                .add_node("cross_encode_rerank", self.run_cross_encode_rerank)
                .add_node("pruning", self.run_pruning)
                .add_node("generate", self.run_generate)
                .add_node(
                    "faster_retrieve_and_rerank", self.run_faster_retrieve_and_rerank
                )
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
        self.latency = {
            k: "0.00 s"
            for k in [
                "analyze_query",
                "retrieve",
                "pruning",
                "generate",
                "re_ranking",
                "faster_retrieve_and_rerank",
                "overall",
            ]
        }
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
    def run_analyze_query(self, state: State, nome_azienda: str = ""):
        """Returns fixed optimized query passed initially

        Args:
            opt_query_predef (str): Query of the form: "<query> AZIENDA:<>"
        """

        t1 = time.time()

        logger.debug(
            f"\nInputs to analyze_query:\nquestion: {state['question']}\nazienda_name_records: {self.azienda_name_records}\nuse_google_api: {self.use_google_api}\n"
        )
        response = state["query"]

        self.latency["analyze_query"] = f"{time.time() - t1:.3f} s"
        # save query for testing
        self.optimized_query = {
            "query": response.query,
            "azienda": response.azienda,
        }  ###

        logger.debug(f"\nOutput of analyze_query:\n'query': {response}\n\n")
        return {"query": response}

    # Async version
    async def arun_analyze_query(self, state: State, nome_azienda: str = ""):
        """Returns fixed optimized query passed initially

        Args:
            state (State): State dict containing pre-defined query for testing.
        """

        t1 = time.time()

        logger.debug(
            f"\nInputs to (async) analyze_query:\nquestion: {state['question']}\nazienda_name_records: {self.azienda_name_records}\nuse_google_api: {self.use_google_api}\n"
        )
        # to mimic real response call
        wait_time = random.randint(1, 3)
        await asyncio.sleep(wait_time)
        response = state["query"]

        self.latency["analyze_query"] = f"{time.time() - t1:.3f} s"
        self.optimized_query = {
            "query": response.query,
            "azienda": response.azienda,
        }  ###

        logger.debug(f"\nOutput of (async) analyze_query:\n'query': {response}\n\n")
        return {"query": response}

    # --------- RETRIEVER ----------

    # Async version
    async def arun_retrieve(self, state: State):

        t1 = time.time()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(
            f"\nInputs to (async) retrieve:\nquery: {query}\nazienda: {azienda}\n"
        )
        output_retrieve = {
            "context": state["context"],
            "retrieved_docs_ids": state["retrieved_docs_ids"],
            "retrieved_docs_texts": state["retrieved_docs_texts"]
        }

        retrieved_docs = output_retrieve.get("context", [])
        self.retrieved_docs_ids = output_retrieve.get("retrieved_docs_ids", {})
        self.retrieved_docs_texts = output_retrieve.get("retrieved_docs_texts", {})

        self.latency["retrieve"] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) retrieve:\n'context': {retrieved_docs}\n\n")
        return {"context": retrieved_docs} 
    
    # --------- RE-RANKER ----------

    # Async version
    async def arun_cross_encode_rerank(self, state: State) -> Dict[str, List[Document]]:
        """Async implementation of run_cross_encode_rerank"""

        t1 = time.time()

        logger.debug(
            f"\nInputs to (async) cross_encode_rerank:\ncontexts: {state['context']}\nquestion: {state['question']}\n"
        )

        # to mimic real response call
        wait_time = random.randint(1, 3)
        await asyncio.sleep(wait_time)
        re_ranker_output = {
            "context": state["re_ranked_context"],
            "re_ranked_docs_ids": state["re_ranked_docs_ids"],
            "re_ranked_docs_texts": state["re_ranked_docs_texts"],
        }

        re_ranked_docs = re_ranker_output.get("context", [])
        self.re_ranked_docs_ids = re_ranker_output.get("re_ranked_docs_ids", {})
        self.re_ranked_docs_texts = re_ranker_output.get("re_ranked_docs_texts", {})

        self.latency["re_ranking"] = f"{time.time() - t1:.3f} s"

        logger.debug(
            f"\nOutput of (async) cross_encode_rerank:\n'context': {re_ranked_docs}\n\n"
        )
        return {"context": re_ranked_docs}  ### re_ranked_docs_parent

    # --------- GENERATOR ----------

    # Async version
    async def arun_generate(self, state: State):
        """Async version run_generate"""
        t1 = time.time()

        logger.debug(
            f"\nInputs to (async) generate:\nquestion: {state['question']}\ncontexts:\n {state['context']}\n"
        )

        # to mimic real response call
        wait_time = random.randint(2, 4)
        await asyncio.sleep(wait_time)
        answer = state['answer']

        self.latency["generate"] = f"{time.time() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) generate:\n'answer': {answer}\n\n")
        # return {"answer": answer}
        return {k: v for k, v in state.items()}

     # =========== XXX ====================

    # other methods
    # Async version
    async def aget_response(self, query: str, state_info: dict={}, nome_azienda: str="") -> State:
        """Async version of get_response"""
        try:
            t1 = time.time()
            response = await self.graph.ainvoke(state_info)  # type: ignore
            self.latency["overall"] = f"{time.time() - t1:.3f} s"
            # return response["answer"]
            return State(**response)
        except Exception as e:
            print(f"Exception: {e}")
            logger.exception(f"Exception: {e}")
            return State(**{})


class DummyExtractInfo(extract_info_module.ExtractInfo):
    """Same class as ExtractInfo with only change: we pass all values of 'State' obj to RAG pipeline"""
    
    def __init__(
        self,
        llm: OllamaLLM,
        extractor_graph: DummyRAGPipeline,
        nome_azienda: str,
        sub_modules: List = [],
        use_google_api: bool = False
    ) -> None:
        super().__init__(llm, extractor_graph, nome_azienda, sub_modules, use_google_api)
        
        self.state_infos = pre_def_info_for_test()
        self.extractor_graph = extractor_graph
    
    # Only changed methods

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

    # ============== Async versions ========================
    async def aextract_info(self):

        async def _extract_info_wrapper(m):
            logger.info(f"## \t Extracting info for moudle {m}: ")

            # Re-set time
            self.extractor_graph.reset_latency()
            logger.debug("Nome Azienda in : %s", self.nome_azienda.upper())  ###
            q = m.question + f" Nome della società: {self.nome_azienda.upper()}"
            state_info = self.state_infos[m.__name__]
            # answer = await self.extractor_graph.aget_response(q, state_info=state_info) # type: ignore
            answer_state = await self.extractor_graph.aget_response(q, state_info=state_info) # type: ignore
            time_consumed = self.extractor_graph.latency

            t1 = time.perf_counter()

            # structure answer only if llm found the response
            if answer_state["answer"] not in ("", "Non ho trovato la risposta nei documenti forniti"):
                formatted_output = await self.aextract_sub_module(m.question, answer_state["answer"], m)
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
            logger.debug("Answer: \t %s", answer_state["answer"])  ###

            ## Save retrieved contexts and optimized query for testing
            self.optimized_query[name_sub_module] = (
                # self.extractor_graph.optimized_query
                answer_state["query"].model_dump()
            )  ###
            self.ori_contexts[name_sub_module] = (
                # self.extractor_graph.retrieved_docs_ids
                answer_state["retrieved_docs_ids"]
            )  ###
            self.ori_contexts_texts[name_sub_module] = (
                # self.extractor_graph.retrieved_docs_texts
                answer_state["retrieved_docs_texts"]
            )  ###
            try:
                self.re_ranked_contexts[name_sub_module] = (
                    # self.extractor_graph.re_ranked_docs_ids
                    answer_state["re_ranked_docs_ids"]
                )  ###
            except AttributeError as e:
                logger.error("re_ranked_docs_ids not found")
                logger.exception(e)
                self.re_ranked_contexts[name_sub_module] = {}  # []
            try:
                self.re_ranked_contexts_texts[name_sub_module] = (
                    # self.extractor_graph.re_ranked_docs_texts
                    answer_state["re_ranked_docs_texts"]
                )  ###
            except AttributeError as e:
                logger.error("re_ranked_docs_texts not found")
                logger.exception(e)
                self.re_ranked_contexts_texts[name_sub_module] = {}  ###

            self.rag_qa[name_sub_module] = {"Q": q, "A": answer_state["answer"]}  ###

            time_consumed["overall"] = (
                f"{float(time_consumed.get('overall', '0 s')[:-2]) + float(time_consumed.get('extract_sub_module', '0 s')[:-2])} s"  ###
            )
            self.run_times[name_sub_module] = time_consumed  ###

            logger.debug("Time consumed on %s: %s", name_sub_module, time_consumed)

        await asyncio.gather(*[_extract_info_wrapper(m) for m in self.sub_modules])


# MONKEY-PATCH: Replace extract_info_module's 'ExtractInfo' with the dummy version "DummyExtractInfo".
# Why: The original "ExtractInfo" makes real, time-consuming API calls. 
# Overriding it here ensures our test runs quickly and doesn't rely on external network connectivity or hit API rate limits.
# For quick testing
extract_info_module.ExtractInfo = DummyExtractInfo

async def test_aextract_and_save_all_info():

    # Define Inits
    llm = OllamaLLM(llm_model="gemma-4-26b-a4b-it")
    rag_obj = DummyRAGPipeline(
        azienda_name_records=["2kind srl"],
        run_async = True,
        use_google_api = False,
    )
    nome_azienda = "2kind srl"
    filename = "8048909650002"
    use_google_api = True
    save_dir = SCRIPT_DIR.parent / "outputs/test_extract_info"
    save_dir.mkdir(parents=True, exist_ok=True)

    await aextract_and_save_all_info(
        rag_pipeline=rag_obj, # type: ignore
        nome_delle_aziende=[(filename, nome_azienda)],
        llm_json=llm,
        save_dir=str(save_dir),
        use_google_api=use_google_api,
    )
    

# =============== FUNCTIONS TO TEST PROBLEM FOR ASYNC PIPELINE ======================================
# PROBLEM:
# The async call is currently populating the exact same data (optimized_query, retrieved_docs_ids, re_ranked_docs_ids, ...) across queries.

async def async_RAG_pipeline_2_queries(
    rag_obj: DummyRAGPipeline,
    infos_to_extract: dict[str,State]
):

    # state_info_names = [x for x in infos_to_extract.keys()]
    answers = {}
    ori_contexts = {}  ###
    re_ranked_contexts = {}  ###
    rag_qa = {}### # Stores question and answer by the rag pipeline for each submodule
    run_times = {}  ### # Store run time for each submodule
    ori_contexts_texts = {}  ###
    re_ranked_contexts_texts = {}  ###
    optimized_query = {}
    temp_filed = {}

    async def _async_RAG_single_query(state_info_name: str, state_info: State):
        rag_obj.reset_latency()
        # curr_result = {}
        q = state_info['question']
        # answer = await rag_obj.aget_response(q, state_info=state_info) # type: ignore
        rag_output_state = await rag_obj.aget_response(q, state_info=state_info) # type: ignore

        # Save info
        # curr_result["answer"] = answer
        # curr_result["run_time"] = rag_obj.latency
        # curr_result['ori_contexts'] = rag_obj.retrieved_docs_ids
        # curr_result["ori_contexts_texts"] = rag_obj.retrieved_docs_texts
        # curr_result["re_ranked_contexts"] = rag_obj.re_ranked_docs_ids
        # curr_result["re_ranked_contexts_texts"] = rag_obj.re_ranked_docs_texts
        # curr_result["rag_qa"] = {"Q": q, "A": answer}
        # curr_result["optimized_query"] = rag_obj.optimized_query
        
        
        answers[state_info_name] = rag_output_state['answer'] # answer
        run_times[state_info_name] = rag_obj.latency #
        ori_contexts[state_info_name] = rag_output_state['retrieved_docs_ids'] # rag_obj.retrieved_docs_ids
        ori_contexts_texts[state_info_name] = rag_output_state['retrieved_docs_texts'] # rag_obj.retrieved_docs_texts
        re_ranked_contexts[state_info_name] = rag_output_state['re_ranked_docs_ids'] # rag_obj.re_ranked_docs_ids
        re_ranked_contexts_texts[state_info_name] = rag_output_state['re_ranked_docs_texts'] # rag_obj.re_ranked_docs_texts
        rag_qa[state_info_name] = {"Q": q, "A": rag_output_state['answer']} # {"Q": q, "A": answer}
        optimized_query[state_info_name] = rag_output_state['query'] # rag_obj.optimized_query

        
        # return curr_result


    # Run queries async
    # outputs = await asyncio.gather(*[_async_RAG_single_query(state_info_name, state_info) for state_info_name, state_info in infos_to_extract.items()])
    # return {k: v for k, v in zip(state_info_names, outputs)}
    await asyncio.gather(*[_async_RAG_single_query(state_info_name, state_info) for state_info_name, state_info in infos_to_extract.items()])
    return {
        "answers": answers,
        "run_times": run_times,
        "ori_contexts": ori_contexts,
        "ori_contexts_texts": ori_contexts_texts,
        "re_ranked_contexts": re_ranked_contexts,
        "re_ranked_contexts_texts": re_ranked_contexts_texts,
        "rag_qa": rag_qa,
        "optimized_query": optimized_query,
        "temp_filed": temp_filed
    }

async def test_async_RAG_pipeline_2_queries():

    # Read pre-defined answers
    infos = pre_def_info_for_test() # state_CapitaleSociale, state_DataChiusuraEsercizio

    # Initialize RAGPipeline and info to extract
    llm = OllamaLLM(llm_model="gemma-4-26b-a4b-it")
    rag_obj = DummyRAGPipeline(
        azienda_name_records=["2kind srl"],
        run_async = True,
        use_google_api = False,
    )
    infos_to_extract = {
        "CapitaleSociale": infos["CapitaleSociale"],
        "DataChiusuraEsercizio": infos["DataChiusuraEsercizio"]
    }

    # Test
    results = await async_RAG_pipeline_2_queries(rag_obj, infos_to_extract)

    # for k in infos_to_extract.keys():
    #     print(k, ":\n")
    #     for r_type, r in results[k].items():
    #         print(f"{r_type}:\t {r}")
    #     print("\n\n")

    for r_type, r in results.items():
        print(f"{r_type}:\t {r}")
        print("\n\n")

# ====================================== XXX =========================================================


if __name__ == "__main__":
    pass
    # asyncio.run(test_aextract_and_save_all_info())
    # asyncio.run(test_async_RAG_pipeline_2_queries())


    # TODO:
    # Test in actual extract_info