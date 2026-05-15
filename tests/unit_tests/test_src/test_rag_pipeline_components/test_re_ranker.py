import json

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever

import rag_info_extractor.rag_pipeline_components.re_ranker as re_ranker


def _doc(chunk_id, parent_id, text):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "parent_id": parent_id})


def test_tokens_normalizes_and_extracts_alnum_words():
    assert re_ranker._tokens("Ciao, mondo! 123 è OK.") == ["ciao", "mondo", "123", "è", "ok"]


def test_span_density_returns_zero_for_short_query_tokens_only():
    assert re_ranker._span_density("a io tu", "qualunque testo") == 0.0


def test_span_density_computes_unique_query_coverage():
    density = re_ranker._span_density("Policy obblighi policy", "Gli obblighi sono previsti")
    assert density == 0.5


def test_query_type_flags_detects_multiple_cues():
    flags = re_ranker._query_type_flags("How compare policies based on prior evidence")
    assert flags["is_how_why"] is True
    assert flags["is_policy_legal"] is True
    assert flags["is_compare"] is True
    assert flags["has_multi_hop_cues"] is True
    assert flags["is_long"] is False


def test_query_type_flags_detects_long_query():
    long_q = " ".join(["word"] * 18)
    flags = re_ranker._query_type_flags(long_q)
    assert flags["is_long"] is True


def test_similarity_margin_edge_and_normal_case():
    assert re_ranker._similarity_margin([]) == 0.0
    assert re_ranker._similarity_margin([0.0, -0.1]) == 0.0
    assert re_ranker._similarity_margin([10.0, 9.0, 8.0], k_ref=3) == 0.2


def test_cross_encode_rerank_returns_empty_for_no_context():
    out = re_ranker.cross_encode_rerank(
        contexts=[],
        question="q",
        doc_store_large_chunks_path=None,
    )
    assert out["context"] == []
    assert out["re_ranked_docs_ids"] == {}
    assert out["re_ranked_docs_texts"] == {}
    assert out["re_rank_debug"] == {}


def test_cross_encode_rerank_selects_by_score_threshold(monkeypatch):
    docs = [
        _doc(1, 10, "aaa first text"),
        _doc(2, 11, "bbb second text"),
        _doc(3, 12, "ccc third text"),
    ]
    monkeypatch.setattr(re_ranker, "call_reranker_service", lambda query, documents: [0.2, 0.9, 0.3])

    out = re_ranker.cross_encode_rerank(
        contexts=docs,
        question="simple question",
        doc_store_large_chunks_path=None,
        k_min=1,
        k_max=2,
        rel_thresh=0.5,
        use_parent_heuristics=False,
        save_full_chunks=True,
    )

    assert [d.metadata["chunk_id"] for d in out["context"]] == [2]
    assert out["re_ranked_docs_ids"]["children"] == [2]
    assert out["re_ranked_docs_ids"]["parents"] == []
    assert out["re_rank_debug"]["selected_k"] == 1
    assert out["re_rank_debug"]["heuristics"]["need_parent"] is True


def test_cross_encode_rerank_promotes_parents_from_store(monkeypatch, tmp_path):
    docs = [
        _doc(1011, 101, "child one text"),
        _doc(1012, 101, "child two text"),
        _doc(2021, 202, "child three text"),
    ]
    monkeypatch.setattr(re_ranker, "call_reranker_service", lambda query, documents: [0.9, 0.8, 0.7])

    page_content_dir = tmp_path / "page_content"
    metadata_dir = tmp_path / "metadata"
    page_content_dir.mkdir()
    metadata_dir.mkdir()

    (page_content_dir / "101").write_text("parent 101 content", encoding="utf-8")
    (metadata_dir / "101").write_text(json.dumps({"chunk_id": 101, "title": "p101"}), encoding="utf-8")
    (page_content_dir / "202").write_text("parent 202 content", encoding="utf-8")
    (metadata_dir / "202").write_text(json.dumps({"chunk_id": 202, "title": "p202"}), encoding="utf-8")

    out = re_ranker.cross_encode_rerank(
        contexts=docs,
        question="short q",
        doc_store_large_chunks_path=str(tmp_path),
        k_min=2,
        k_max=3,
        max_promoted_parents=2,
        use_parent_heuristics=False,
        save_full_chunks=True,
    )

    assert [d.metadata.get("chunk_id") for d in out["context"]] == [101, 202]
    assert out["re_ranked_docs_ids"]["parents"] == [101, 202]
    assert out["re_ranked_docs_ids"]["children"] == []


@pytest.mark.asyncio
async def test_across_encode_rerank_empty_context():
    out = await re_ranker.across_encode_rerank(
        contexts=[],
        question="q",
        doc_store_large_chunks_path=None,
    )
    assert out["context"] == []
    assert out["re_ranked_docs_ids"] == {}
    assert out["re_ranked_docs_texts"] == {}
    assert out["re_rank_debug"] == {}


@pytest.mark.asyncio
async def test_across_encode_rerank_with_parent_heuristics_disabled(monkeypatch):
    docs = [_doc(1, 10, "alpha"), _doc(2, 10, "beta"), _doc(3, 30, "gamma")]

    async def fake_async_rerank(query, documents):
        return [0.4, 0.95, 0.2]

    monkeypatch.setattr(re_ranker, "acall_reranker_service", fake_async_rerank)

    out = await re_ranker.across_encode_rerank(
        contexts=docs,
        question="q",
        doc_store_large_chunks_path=None,
        k_min=1,
        k_max=2,
        rel_thresh=0.5,
        use_parent_heuristics=False,
    )

    assert [d.metadata["chunk_id"] for d in out["context"]] == [2]
    assert out["re_rank_debug"]["selected_k"] == 1


def test_reranker_base_ce_score_calls_service(monkeypatch):
    captured = {}

    def fake_service(query, documents):
        captured["query"] = query
        captured["documents"] = documents
        return [0.11, 0.22]

    monkeypatch.setattr(re_ranker, "call_reranker_service", fake_service)
    model = re_ranker.ReRankerBaseCE()
    scores = model.score([("Q", "doc a"), ("Q", "doc b")])

    assert scores == [0.11, 0.22]
    assert captured["query"] == "Q"
    assert captured["documents"] == ["doc a", "doc b"]


class _FakeCompressionRetriever(VectorStoreRetriever):
    def __init__(self, docs):
        self.docs = docs
        self.invoke_kwargs = None
        self.ainvoke_kwargs = None

    def invoke(self, query, **kwargs):
        self.invoke_kwargs = {"query": query, **kwargs}
        return self.docs

    async def ainvoke(self, query, **kwargs):
        self.ainvoke_kwargs = {"query": query, **kwargs}
        return self.docs


def test_faster_retrieve_and_rerank_dedups_and_applies_filter(monkeypatch):
    docs = [
        Document(page_content="start<JOIN>end", metadata={"chunk_id": 1, "azienda": "A"}),
        Document(page_content="dup should drop", metadata={"chunk_id": 1, "azienda": "A"}),
        Document(page_content="second<JOIN>text", metadata={"chunk_id": 2, "azienda": "A"}),
    ]
    fake_retriever = object()
    fake_ccr = _FakeCompressionRetriever(docs)

    monkeypatch.setattr(re_ranker, "CrossEncoderReranker", lambda model, top_n: object())
    monkeypatch.setattr(
        re_ranker,
        "ContextualCompressionRetriever",
        lambda base_compressor, base_retriever: fake_ccr,
    )

    out = re_ranker.faster_retrieve_and_rerank(
        query="q",
        retriever=fake_retriever,
        azienda="A",
        top_n=4,
        pages_joining_str="<JOIN>",
        save_full_chunks=True,
    )

    assert fake_ccr.invoke_kwargs == {"query": "q", "filter": {"azienda": "A"}, "k": 4}
    assert [d.metadata["chunk_id"] for d in out["context"]] == [1, 2]
    assert out["context"][0].page_content == "start\nend"
    assert out["docs_ids"] == {"parents": [], "children": [1, 2]}
    assert out["docs_texts"]["children"] == ["start\nend", "second\ntext"]


def test_faster_retrieve_and_rerank_without_azienda(monkeypatch):
    docs = [Document(page_content="abcdefghijk", metadata={"chunk_id": 99})]
    fake_ccr = _FakeCompressionRetriever(docs)

    monkeypatch.setattr(re_ranker, "CrossEncoderReranker", lambda model, top_n: object())
    monkeypatch.setattr(
        re_ranker,
        "ContextualCompressionRetriever",
        lambda base_compressor, base_retriever: fake_ccr,
    )

    out = re_ranker.faster_retrieve_and_rerank(query="q", retriever=object(), top_n=1, save_full_chunks=False)
    assert fake_ccr.invoke_kwargs == {"query": "q", "k": 1}
    assert out["docs_texts"]["children"][0] == "abcdefghij ... bcdefghijk"


@pytest.mark.asyncio
async def test_afaster_retrieve_and_rerank_async_path(monkeypatch):
    docs = [
        Document(page_content="first<SEP>chunk", metadata={"chunk_id": 4}),
        Document(page_content="first<SEP>chunk-dup", metadata={"chunk_id": 4}),
        Document(page_content="second<SEP>chunk", metadata={"chunk_id": 5}),
    ]
    fake_ccr = _FakeCompressionRetriever(docs)

    monkeypatch.setattr(re_ranker, "CrossEncoderReranker", lambda model, top_n: object())
    monkeypatch.setattr(
        re_ranker,
        "ContextualCompressionRetriever",
        lambda base_compressor, base_retriever: fake_ccr,
    )

    out = await re_ranker.afaster_retrieve_and_rerank(
        query="aq",
        retriever=object(),
        azienda="ACME",
        top_n=3,
        pages_joining_str="<SEP>",
        save_full_chunks=False,
    )

    assert fake_ccr.ainvoke_kwargs == {"query": "aq", "filter": {"azienda": "ACME"}, "k": 3}
    assert out["docs_ids"]["children"] == [4, 5]
    assert out["docs_texts"]["children"] == ["first\nchun ... rst\nchunk", "second\nchu ... ond\nchunk"]







# ======================================== XXX ========================================
from rag_info_extractor.rag_pipeline.re_ranker import faster_retrieve_and_rerank, afaster_retrieve_and_rerank, across_encode_rerank, cross_encode_rerank
from rag_info_extractor.utils.embedder import HFEmbedder
from langchain_chroma import Chroma
import os
import asyncio



DOC_STORE_LARGE_CHUNKS_PATH = "D:/Documents/Italy/UNIPD/University Acadamico/TESI/project/rag_information_extractor/data/large_chunks_dbs/temp_tests/custom_chunks_2"
VECTOR_STORE_PATH = "D:/Documents/Italy/UNIPD/University Acadamico/TESI/project/rag_information_extractor/data/vector_dbs/temp_tests/custom_chunks_2"
if not os.path.exists(DOC_STORE_LARGE_CHUNKS_PATH):
    raise FileNotFoundError(f"DOC_STORE_LARGE_CHUNKS_PATH not found: {DOC_STORE_LARGE_CHUNKS_PATH}")
if not os.path.exists(VECTOR_STORE_PATH):
    raise FileNotFoundError(f"VECTOR_STORE_PATH not found: {VECTOR_STORE_PATH}")

# Load Vector and Doc store
embedding = HFEmbedder(normalize_embeddings=True)
vector_store = Chroma(embedding_function=embedding,
                    persist_directory=VECTOR_STORE_PATH,
                    collection_name="pdf_chunks")
retriever = vector_store.as_retriever(search_type="similarity",
                                    search_kwargs={'k': 8})

# Test functions
def test_cross_encode_rerank():
    pass

def test_faster_retrieve_and_rerank():
    QUERY = "Agli amministratori spetta il rimborso delle spese?"
    AZIENDA = "2kind srl"
    output = faster_retrieve_and_rerank(
        query = QUERY,
        retriever = retriever,
        azienda = AZIENDA,
        top_n = 4,
        pages_joining_str = None,
        save_full_chunks = False
    )
    for c in output["context"]:
        assert c.metadata.get("azienda") == AZIENDA
    assert output["context"] is not None
    assert output["docs_texts"] is not None
    assert output["docs_ids"].get("parents") == []
    assert output["docs_ids"].get("children") == [16, 18, 11, 10]

async def test_afaster_retrieve_and_rerank():
    QUERY_1 = "Agli amministratori spetta il rimborso delle spese?"
    AZIENDA_1 = "2kind srl"
    QUERY_2 = "Qual è l’ammontare del capitale sociale della società?"
    AZIENDA_2 = "2kind srl"

    tasks = [
        afaster_retrieve_and_rerank(
            query = q,
            retriever = retriever,
            azienda = a,
            top_n = 4,
            pages_joining_str = None,
            save_full_chunks = False
        )
        for q, a in zip([QUERY_1, QUERY_2], [AZIENDA_1, AZIENDA_2])
    ]
    outputs = await asyncio.gather(*tasks)

    for i, output in enumerate(outputs):
        if i == 0:            
            for c in output["context"]:
                assert c.metadata.get("azienda") == AZIENDA_1
            assert output["context"] is not None
            assert output["docs_texts"] is not None
            assert output["docs_ids"].get("parents") == []
            assert output["docs_ids"].get("children") == [16, 18, 11, 10]
        
        if i == 1:
            for c in output["context"]:
                assert c.metadata.get("azienda") == AZIENDA_2
            assert output["context"] is not None
            assert output["docs_texts"] is not None
            assert output["docs_ids"].get("parents") == []
            assert output["docs_ids"].get("children") == [4, 9, 2, 18]
 
if __name__ == "__main__":
    # test_faster_retrieve_and_rerank()
    asyncio.run(test_afaster_retrieve_and_rerank())