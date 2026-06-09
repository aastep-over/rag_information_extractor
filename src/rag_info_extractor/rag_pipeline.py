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

DetectorFactory.seed = 0  # probablistic algo.

import asyncio

# logging relative
import logging
import re

# Python native
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

import aiofiles

from rag_info_extractor.rag_pipeline_components.analyze_query import (
    Search,
    aanalyze_query,
    analyze_query,
)
from rag_info_extractor.rag_pipeline_components.generator import agenerate, generate
from rag_info_extractor.rag_pipeline_components.re_ranker import (
    across_encode_rerank,
    afaster_retrieve_and_rerank,
    cross_encode_rerank,
    faster_retrieve_and_rerank,
)
from rag_info_extractor.rag_pipeline_components.retrieve import aretrieve, retrieve
from rag_info_extractor.utils.apis_connector import (
    acall_pruner_service,
    call_pruner_service,
)

# Import rag parts from modules
from rag_info_extractor.utils.llm_connector import OllamaLLM

# from rag_info_extractor.rag_pipeline_components.verify import verify

logger = logging.getLogger(__name__)


# TODO:
# Aggiungere se query non menziona l'azienda, farlo per ogni azienda con citazione


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    rerank_debug: Dict
    answer: str
    retrieved_docs_ids: Dict[str, List[int]]
    retrieved_docs_texts: Dict[str, List[str]]
    re_ranked_docs_ids: Dict[str, List[int]]
    re_ranked_docs_texts: Dict[str, List[str]]




class RAGPipeline:

    def __init__(
        self,
        db_retriever: VectorStoreRetriever,
        azienda_name_records: List[str],
        llm_model: str,
        doc_store_path: Optional[str],
        pages_joining_str: Optional[str],
        run_async: bool = False,
        use_google_api: bool = False,
    ):
        """
        Args:
            db_retriever (VectorStoreRetriever): retriver function for the vector database.
            azienda_name_records (List[str]): names of all azienda/società for which the record exist in db
            doc_store_path (Optional[str]): path of the folder containing parent/large chunks
        """

        # Initialize/Load the vector db and doc_store and llm
        self.retriever = db_retriever
        self.llm = OllamaLLM(llm_model=llm_model)

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
                .add_node(
                    "faster_retrieve_and_rerank", self.arun_faster_retrieve_and_rerank
                )
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
                .add_node(
                    "faster_retrieve_and_rerank", self.run_faster_retrieve_and_rerank
                )
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

    def reset_latency(self):
        for k in self.latency:
            self.latency[k] = "0.00 s"

    # =========== NODES ====================

    # --------- ANALYZE QUERY ----------
    def run_analyze_query(self, state: State, nome_azienda: str = ""):
        """
        Re-phrase query to improve it

        nome_azienda (str): name of the società for which the information needs to be extracted
        """

        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to analyze_query:\nquestion: {state['question']}\nazienda_name_records: {self.azienda_name_records}\nuse_google_api: {self.use_google_api}\n"
        )
        response: Search = analyze_query(
            question=state["question"],
            llm=self.llm,
            nome_azienda="",
            azienda_name_records=self.azienda_name_records,
            use_google_api=self.use_google_api,
        )
        # ==================== FOR QUICK TESTING =========================================
        # question, optimized_query, azienda_name= re.findall(r"(.*) QUERY: (.*) Nome della società: (.*)", state['question'])[0] # TODO: REMOVE AFTER TESTING
        # state['question'] = question
        # response = Search(query=optimized_query, azienda=azienda_name) # TODO: REMOVE AFTER TESTING
        # =========================   XXX   =========================================

        self.latency["analyze_query"] = f"{time.perf_counter() - t1:.3f} s"
    
        logger.debug(f"\nOutput of analyze_query:\n'query': {response}\n\n")
        return {"query": response}

    # Async version
    async def arun_analyze_query(self, state: State, nome_azienda: str = ""):
        """
        Re-phrase query to improve it

        nome_azienda (str): name of the società for which the information needs to be extracted
        """

        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to (async) analyze_query:\nquestion: {state['question']}\nazienda_name_records: {self.azienda_name_records}\nuse_google_api: {self.use_google_api}\n"
        )
        response: Search = await aanalyze_query(
            question=state["question"],
            llm=self.llm,
            nome_azienda="",
            azienda_name_records=self.azienda_name_records,
            use_google_api=self.use_google_api,
        )

        self.latency["analyze_query"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) analyze_query:\n'query': {response}\n\n")
        return {"query": response}

    # --------- RETRIEVER ----------
    def run_retrieve(self, state: State):

        t1 = time.perf_counter()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(f"\nInputs to retrieve:\nquery: {query}\nazienda: {azienda}\n")
        output_retrieve = retrieve(
            retriever=self.retriever,
            query=query,
            doc_store_large_chunks_path=self.doc_store_path,
            azienda=azienda,
            pages_joining_str=self.pages_joining_str,
            retrieve_parents=False,
            save_full_chunks=False,
        )

        retrieved_docs = output_retrieve.get("context", [])

        self.latency["retrieve"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(f"\nOutput of retrieve:\n'context': {retrieved_docs}\n\n")
        # return {"context": retrieved_docs}  # retrieved_docs_parent
        return {
            "context": retrieved_docs,
            "retrieved_docs_ids": output_retrieve.get("retrieved_docs_ids", {}),
            "retrieved_docs_texts": output_retrieve.get("retrieved_docs_texts", {}),
        }

    # Async version
    async def arun_retrieve(self, state: State):

        t1 = time.perf_counter()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(
            f"\nInputs to (async) retrieve:\nquery: {query}\nazienda: {azienda}\n"
        )
        output_retrieve = await aretrieve(
            retriever=self.retriever,
            query=query,
            doc_store_large_chunks_path=self.doc_store_path,
            azienda=azienda,
            pages_joining_str=self.pages_joining_str,
            retrieve_parents=False,  # TODO: DEFAULT=False
            save_full_chunks=False,
        )

        retrieved_docs = output_retrieve.get("context", [])

        self.latency["retrieve"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) retrieve:\n'context': {retrieved_docs}\n\n")
        # return {"context": retrieved_docs}  # retrieved_docs_parent
        return {
            "context": retrieved_docs,
            "retrieved_docs_ids": output_retrieve.get("retrieved_docs_ids", {}),
            "retrieved_docs_texts": output_retrieve.get("retrieved_docs_texts", {}),
        }

    # --------- PRUNER ----------
    def run_pruning(self, state: State):
        # Execute pruning
        ori_context = [c.page_content for c in state["context"]]
        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to pruning:\nquery: {state['question']}\ndocuments: {ori_context}\n"
        )
        pruned_docs = call_pruner_service(
            query=state["question"], documents=ori_context
        )

        self.latency["pruning"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(f"\nOutput of pruning:\n'context': {pruned_docs}\n\n")

        return {"context": [Document(page_content=c) for c in pruned_docs]}

    # Async version
    async def arun_pruning(self, state: State):
        # Execute pruning
        ori_context = [c.page_content for c in state["context"]]
        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to (async) pruning:\nquery: {state['question']}\ndocuments: {ori_context}\n"
        )
        pruned_docs = await acall_pruner_service(
            query=state["question"], documents=ori_context
        )

        self.latency["pruning"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(f"\nOutput of (async) pruning:\n'context': {pruned_docs}\n\n")
        return {"context": [Document(page_content=c) for c in pruned_docs]}

    # --------- RE-RANKER ----------
    def run_cross_encode_rerank(self, state: State) -> dict:
        """Re-ranks the retrieved docs"""

        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to cross_encode_rerank:\ncontexts: {state['context']}\nquestion: {state['question']}\n"
        )
        re_ranker_output = cross_encode_rerank(
            contexts=state["context"],
            question=state["question"],
            doc_store_large_chunks_path=self.doc_store_path,
            k_min=2,
            k_max=5,
            rel_thresh=0.4,
            max_promoted_parents=3,
            use_parent_heuristics=False,
            save_full_chunks=False,
        )

        re_ranked_docs = re_ranker_output.get("context", [])
        re_ranker_debug_info = re_ranker_output.get("re_rank_debug", {})

        self.latency["re_ranking"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(
            f"\nOutput of cross_encode_rerank:\n'context': {re_ranked_docs}\n\n"
        )
        
        return {
            "context": re_ranked_docs,
            "re_ranked_docs_ids": re_ranker_output.get("re_ranked_docs_ids", {}),
            "re_ranked_docs_texts": re_ranker_output.get("re_ranked_docs_texts", {}),
            "rerank_debug": re_ranker_debug_info,
        }

    # Async version
    async def arun_cross_encode_rerank(self, state: State) -> dict:
        """Async implementation of run_cross_encode_rerank"""

        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to (async) cross_encode_rerank:\ncontexts: {state['context']}\nquestion: {state['question']}\n"
        )
        re_ranker_output = await across_encode_rerank(
            contexts=state["context"],
            question=state["question"],
            doc_store_large_chunks_path=self.doc_store_path,
            k_min=2,
            k_max=5,
            rel_thresh=0.4,
            max_promoted_parents=3,
            use_parent_heuristics=False,
            save_full_chunks=False,
        )

        re_ranked_docs = re_ranker_output.get("context", [])
        re_ranker_debug_info = re_ranker_output.get("re_rank_debug", {})

        self.latency["re_ranking"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(
            f"\nOutput of (async) cross_encode_rerank:\n'context': {re_ranked_docs}\n\n"
        )

        return {
            "context": re_ranked_docs,
            "re_ranked_docs_ids": re_ranker_output.get("re_ranked_docs_ids", {}),
            "re_ranked_docs_texts": re_ranker_output.get("re_ranked_docs_texts", {}),
            "rerank_debug": re_ranker_debug_info,
        }

    # --------- FAST RETRIEVER + RE-RANKER ----------
    def run_faster_retrieve_and_rerank(self, state: State):
        """Retrieve + Re-rank + Compress context"""
        t1 = time.perf_counter()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(
            f"\nInputs to faster_retrieve_and_rerank:\nquery: {query}\nazienda: {azienda}\n"
        )
        output = faster_retrieve_and_rerank(
            query=query,
            retriever=self.retriever,
            azienda=azienda,
            top_n=6,
            pages_joining_str=self.pages_joining_str,
            save_full_chunks=True,  # default=False
        )

        docs = output.get("context", [])

        self.latency["faster_retrieve_and_rerank"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(f"\nOutput of faster_retrieve_and_rerank:\n'context': {docs}\n\n")
        # return {"context": docs}
        return {
            "context": docs,
            "re_ranked_docs_ids": output.get("docs_ids", {}),
            "re_ranked_docs_texts": output.get("docs_texts", {}),
        }

    # Async version
    async def arun_faster_retrieve_and_rerank(self, state: State):
        """Async version of run_faster_retrieve_and_rerank"""
        t1 = time.perf_counter()
        query = state["query"].query
        azienda = state["query"].azienda

        logger.debug(
            f"\nInputs to (async) faster_retrieve_and_rerank:\nquery: {query}\nazienda: {azienda}\n"
        )
        output = await afaster_retrieve_and_rerank(
            query=query,
            retriever=self.retriever,
            azienda=azienda,
            top_n=6,
            pages_joining_str=self.pages_joining_str,
            save_full_chunks=True,  # default=False
        )

        docs = output.get("context", [])
        self.latency["faster_retrieve_and_rerank"] = f"{time.perf_counter() - t1:.3f} s"

        logger.debug(
            f"\nOutput of (async) faster_retrieve_and_rerank:\n'context': {docs}\n\n"
        )
        # return {"context": docs}
        return {
            "context": docs,
            "re_ranked_docs_ids": output.get("docs_ids", {}),
            "re_ranked_docs_texts": output.get("docs_texts", {}),
        }

    # --------- GENERATOR ----------
    def run_generate(self, state: State):

        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to generate:\nquestion: {state['question']}\ncontexts:\n {state['context']}\n"
        )
        answer = generate(
            question=state["question"],
            contexts=state["context"],
            llm=self.llm,
            contexts_sep="||",
            use_google_api=self.use_google_api,
        )

        # detect if answer in italiano, if not regenerate (1 try max to avoid constant loop)
        if detect(answer) != "it":
            print("Regenerating answer....")  ###
            answer = generate(
                question=state["question"],
                contexts=state["context"],
                llm=self.llm,
                contexts_sep="||",
                additional_prompt="Rispondere sempre in Italiano.",
                use_google_api=self.use_google_api,
            )

        self.latency["generate"] = f"{time.perf_counter() - t1:.3f} s"
        state['answer'] = answer

        logger.debug(f"\nOutput of generate:\n'answer': {answer}\n\n")

        return {k: v for k, v in state.items()}

    # Async version
    async def arun_generate(self, state: State):
        """Async version run_generate"""
        t1 = time.perf_counter()

        logger.debug(
            f"\nInputs to (async) generate:\nquestion: {state['question']}\ncontexts:\n {state['context']}\n"
        )
        answer = await agenerate(
            question=state["question"],
            contexts=state["context"],
            llm=self.llm,
            contexts_sep="||",
            use_google_api=self.use_google_api,
        )

        # detect if answer in italiano, if not regenerate (1 try max to avoid constant loop)
        if detect(answer) != "it":
            print("Regenerating answer....")  ###
            answer = await agenerate(
                question=state["question"],
                contexts=state["context"],
                llm=self.llm,
                contexts_sep="||",
                additional_prompt="Rispondere sempre in Italiano.",
                use_google_api=self.use_google_api,
            )

        self.latency["generate"] = f"{time.perf_counter() - t1:.3f} s"
        state['answer'] = answer
        logger.debug(f"\nOutput of (async) generate:\n'answer': {answer}\n\n")
        
        return {k: v for k, v in state.items()}

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
    def get_response(self, query: str, nome_azienda: str = "") -> State:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """

        try:
            t1 = time.perf_counter()
            response = self.graph.invoke({"question": query})  # type: ignore
            self.latency["overall"] = f"{time.perf_counter() - t1:.3f} s"
            return State(**response)
        except Exception as e:
            print(f"Exception: {e}")
            return State(**{})

    # Async version
    async def aget_response(self, query: str, nome_azienda: str = "") -> State:
        """Async version of get_response"""
        try:
            t1 = time.perf_counter()
            response = await self.graph.ainvoke({"question": query})  # type: ignore
            self.latency["overall"] = f"{time.perf_counter() - t1:.3f} s"
            return State(**response)
        except Exception as e:
            print(f"Exception: {e}")
            logger.exception(f"Exception: {e}")
            return State(**{})

    def save_DAG_diagram(self, directory: str = ""):
        if directory:
            filename = f"{directory}/rag_pipeline_components.png"
        else:
            filename = "rag_pipeline_components.png"
        try:
            dag_img = self.graph.get_graph().draw_mermaid_png(max_retries=5)
        except:
            logger.exception("Error! DAG for the RAG Pipeline not generated")
        else:
            with open(filename, "wb") as png:
                png.write(dag_img)

    # Async version
    async def asave_DAG_diagram(self, directory: str = ""):
        if directory:
            filename = f"{directory}/rag_pipeline_components.png"
        else:
            filename = "rag_pipeline_components.png"
        try:
            dag_img = self.graph.get_graph().draw_mermaid_png(max_retries=5)
        except:
            logger.exception("Error! DAG for the RAG Pipeline not generated")
        else:
            async with aiofiles.open(filename, "wb") as png:
                await png.write(dag_img)
