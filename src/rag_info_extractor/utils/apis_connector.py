from pydantic import BaseModel

import requests, httpx

from typing import List, Tuple, Any
from dotenv import load_dotenv
import yaml
import os

from rag_info_extractor.utils.load_config import cfgs




# Load configs and environment vars
cfgs = cfgs.get("args", {})
BASE_DIR = cfgs.get("BASE_DIR")
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Load Re-Ranker API
RERANKER_API = os.getenv("RERANKER_API", "")
def call_reranker_service(query: str, documents: List[str]) -> List[float] | None:
    """Chiama il servizio Re-ranker esterno."""
    payload = {
        "query": query,
        "documents": documents
    }
    
    try:
        response = requests.post(
            RERANKER_API, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status() # Launches an exception for error status
        
        result = response.json()
        return result['scores']
        
    except requests.exceptions.RequestException as e:
        print(f"Errore nella chiamata al servizio API: {e}")
        return None

async def acall_reranker_service(query: str, documents: List[str]) -> List[float] | None:
    """Chiama il servizio Re-ranker esterno."""
    payload = {
        "query": query,
        "documents": documents
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RERANKER_API,
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
        response.raise_for_status()

        result = response.json()
        return result['scores']

    except httpx.HTTPStatusError as e:
        print(f"(async) Errore nella chiamata al servizio API: {e}")
        return None


# Load Pruner API
PRUNER_API = os.getenv("PRUNER_API", "")
def call_pruner_service(query: str, documents: List[str]) -> List[str]:
    """Chiama il servizio Pruner esterno."""
    payload = {
        "query": query,
        "documents": documents
    }
    
    try:
        response = requests.post(
            PRUNER_API, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status() # Launches an exception for error status
        
        result = response.json()
        return result['pruned_docs']
        
    except requests.exceptions.RequestException as e:
        print(f"Errore nella chiamata al servizio API: {e}")
        return []

async def acall_pruner_service(query: str, documents: List[str]) -> List[str]:
    """Chiama il servizio Pruner esterno."""
    payload = {
        "query": query,
        "documents": documents
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                PRUNER_API,
                json=payload, 
                headers={"Content-Type": "application/json"}
            )

        response.raise_for_status()
        result = response.json()
        return result['pruned_docs']

    except httpx._exceptions.RequestError as e:
        print(f"Errore nella chiamata al servizio API: {e}")
        return []


# Load Embedder API
EMBEDDING_API = os.getenv("EMBEDDING_API", "")
def call_embedder_service(texts: List[str], **encode_kwargs) -> List[List[float]]:
    """Chiama il servizio di Embedder esterno."""
    payload = {
        "texts": texts,
        "encode_kwargs": encode_kwargs
    }
    
    try:
        response = requests.post(
            EMBEDDING_API, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status() # Launches an exception for error status
        
        result = response.json()
        return result['embeddings']
        
    except requests.exceptions.RequestException as e:
        print(f"Errore nella chiamata al servizio API: {e}")
        return []

async def acall_embedder_service(texts: List[str], **encode_kwargs) -> List[List[float]]:
    """Chiama il servizio di Embedder esterno."""
    payload = {
        "texts": texts,
        "encode_kwargs": encode_kwargs
    }   

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EMBEDDING_API,
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
        response.raise_for_status()
        result = response.json()
        return result['embeddings']

    except httpx._exceptions.RequestError as e:
        print(f"Errore nella chiamata al servizio API: {e}")
        return []



if __name__ == "__main__":

    # Esempio
    user_query = "Qual è la durata della società (fino a quale data)?"
    retrieved_docs = [
        "BILANCIO E UTILIArt.23.- Gli utili netti, in base a delibera assembleare, sono ripartiti comesegue:- il cinque per cento (5%) sarà destinato alla riserva legale fino alraggiungimento dell'importo pari al venti per cento del capitale sociale;- la rimanenza è ripartita fra i soci in proporzione delle rispettive quote dicapitale, salvo che essi non decidano diversamente.",
        "Art.3.- La durata della società è fissata fino al trentuno dicembreduemilasessanta.Con delibera dell'Assemblea dei soci, potrà essere sciolta anticipatamenteo prorogata.",
        'Art.19. Agli Amministratori spetta, oltre al rimborso delle spesesostenute in ragione del loro ufficio, un compenso eventuale determinato daisoci.'
    ]   

    # # Re-ranker test
    # scores = call_reranker_service(user_query, retrieved_docs)

    # if scores:
    #     print(f"\nQuery: {user_query}")
    #     print(f"Punteggi di Re-ranking: {scores}")

    #     # zip the retrieved documents with their corresponding scores for sorting
    #     ranked_docs = sorted(
    #         zip(retrieved_docs, scores), 
    #         key=lambda x: x[1], 
    #         reverse=True
    #     )
    #     print("\nDocumenti Ordinati:")
    #     for doc, score in ranked_docs:
    #         print(f"  - Score {score:.4f}: {doc}")

    
    
    # Pruner Test
    pruned_texts = call_pruner_service(user_query, retrieved_docs)
    print(pruned_texts)

    # # Embedder Test
    # embeddings = call_embedder_service([user_query])
    # print(embeddings)