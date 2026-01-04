from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from transformers import AutoModel

from rag_info_extractor import embedding_server

print("Ciao!")
# # Load reranker models
# RERANKER_MODEL = "D:/Users/yye7607/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# re_ranker = CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512)
# print("Loaded Re-ranker")

# # Load faster rerank model
# EMBEDDING_MODEL_NAME = "D:/Users/yye7607/.hf_models/embedding_models/e5-large-instruct"
# embedding_func = HuggingFaceEmbeddings(
#     model_name = EMBEDDING_MODEL_NAME, # HuggingFace embedding model
#     encode_kwargs = {"normalize_embeddings": True}
# )
# vector_store = Chroma(
#     embedding_function = embedding_func,
#     persist_directory = "D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/data/vector_dbs/temp/custom_chunks",
#     collection_name = "pdf_chunks"
# )

# RERANKER_MODEL = "D:/Users/yye7607/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# fast_reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL), top_n=8)
# retriever = vector_store.as_retriever()
# compression_retriever = ContextualCompressionRetriever(
#             base_compressor=fast_reranker, base_retriever=retriever
#             )
# print("Loaded fASTER Re-ranker")


# # Load pruner model
# PRUNER_MODEL = "D:/Users/yye7607/.cache/huggingface/hub/models--naver--xprovence-reranker-bgem3-v1/snapshots/dd707795cc5a6ef5570f9be5282dab615fb7373d" 
# pruner = AutoModel.from_pretrained(
#     PRUNER_MODEL,
#     trust_remote_code=True,
#     local_files_only=True
# )            
# print("Loaded Pruner")


# run and print scores
query = "Qual è la durata della società (fino a quale data)?"
context_candidates = [
    "\n        Art.3.- La durata della società è fissata fino al trentuno dicembre\n        duemilasessanta.\n        Con delibera dell'Assemblea dei soci, potrà essere sciolta anticipatamente\n        o prorogata.\n        ",
    "\n        BILANCIO E UTILI\n        Art.23.- Gli utili netti, in base a delibera assembleare, sono ripartiti come\n        segue:\n        - il cinque per cento (5%) sarà destinato alla riserva legale fino al\n        raggiungimento dell'importo pari al venti per cento del capitale sociale;\n        - la rimanenza è ripartita fra i soci in proporzione delle rispettive quote di\n        capitale, salvo che essi non decidano diversamente.\n        ",
    "\n        L'Assemblea deve essere convocata almeno una volta l'anno per\n        l'approvazione del bilancio, entro centoventi giorni dalla chiusura\n        dell'esercizio sociale, oppure ove la società sia tenuta alla redazione del\n        bilancio consolidato ovvero quando lo richiedano particolari esigenze\n        relative alla struttura ed all’oggetto della società, entro centottanta giorni\n        dalla sopradetta chiusura; in questi casi gli amministratori segnalano nella\n        relazione prevista dall’art. 2428 del codice civile le ragioni della dilazione.\n        ",

]


# # re-ranker
# pairs = [(query, c) for c in context_candidates]
# scores = re_ranker.predict(pairs, batch_size=2, show_progress_bar=False)
# scores_with_idxs = [(i, round(score, 3)) for i, score in enumerate(scores)]
# print("\n", "-"*40, " RE-RANKER output: ", "-"*40, "\n")
# print("RE-RANKER: ", scores_with_idxs)


# # faster re-ranker + retriever
# docs_retrieved = compression_retriever.invoke(query)
# print("\n", "-"*40, " FASTER re-ranker + retriever output: ", "-"*40, "\n")
# print("\n\n".join(d.page_content for d in docs_retrieved))


# # Pruner
# pruned_contexts = pruner.process([query], [context_candidates], reorder=True, top_k=5, threshold=0.05).get("pruned_context")
# print("\n", "-"*40, " PRUNER: ", "-"*40, "\n")
# for c in pruned_contexts[0]:
#     print("\n\n", c)



