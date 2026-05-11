import os
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import argparse
import torch

# Logging
import logging
from rag_info_extractor.utils.common_logging import configure_logging
logger = logging.getLogger(__name__)

from rag_info_extractor.utils.load_config import cfgs

# --- Global Config ---
cfgs = cfgs.get("args", {})
BASE_DIR = cfgs.get("BASE_DIR")
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Path to EMBEDDING Model Directory/ name of the model
EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_NAME_ENV = EMBEDDING_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
MODEL_NAME_OR_PATH = os.environ.get(EMBEDDING_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME) # Load model from local path if already downloaded
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Input Data Structure ---
class EmbedRequest(BaseModel):
    """Schema per la richiesta di embedding."""
    texts: List[str] = Field(..., description="List of texts to be embedded.")
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""

class EmbedResponse(BaseModel):
    """Schema per la risposta di embedding."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors(in a list)")

# --- Initialize server and model ---

embedding_model = None

# @app.on_event("startup")
@asynccontextmanager
async def load_model(app: FastAPI):
    """
    Funzione eseguita all'avvio dell'applicazione.
    Carica il modello in memoria RAM/VRAM una sola volta.
    """
    global embedding_model
    try:
        print(f"🔄 Caricamento del modello Re-ranker: {MODEL_NAME_OR_PATH}")
        # Load the Cross-Encoder model from path
        embedding_model = SentenceTransformer(
            MODEL_NAME_OR_PATH,
            device=DEVICE,
            trust_remote_code=True,
            local_files_only=True,
            config_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"🖥️  Using device: {DEVICE.upper()}")
        logger.info("✅ Modello caricato con successo.")
    except Exception as e:
        logger.error(f"❌ Errore durante il caricamento del modello: {e}")
        raise HTTPException(status_code=500, detail=f"Impossibile caricare il modello: {e}")
    
    yield

    # Clear value of pruner
    embedding_model = None


app = FastAPI(
    title="RAG Embedding API",
    description="Servizio dedicato al caricamento del modello dell'embedding e fare gli embedding.",
    lifespan=load_model
)

# --- Endpoint API ---
@app.post(
    "/embed", 
    response_model=EmbedResponse, 
    summary="Crea gli embedding di una lista di documenti."
)
def encode(request: EmbedRequest):
    """
    Prende una query e una lista di documenti, e restituisce i punteggi di pertinenza.
    """
    if embedding_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Il modello non è ancora stato caricato o non è disponibile."
        )
    
    # 1. Esegui l'inferenza
    # Il metodo predict è thread-safe per Sentence-Transformers
    try:
        embeddings = embedding_model.encode(
            request.texts,
            batch_size=32 if DEVICE == 'cuda' else 4,
            **request.encode_kwargs
        )
    except Exception as e:
        # Gestione di errori di inferenza (es. batch size troppo grande)
        raise HTTPException(
            status_code=500, 
            detail=f"Errore durante l'inferenza del modello: {e}"
        )

    # 3. Restituisci i risultati
    return EmbedResponse(embeddings=[x.tolist() for x in embeddings])




# --- Avvio del Server ---
if __name__ == "__main__":
    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\re_ranker_api.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Launching EMBEDDING API (RAG_INFORMATION_EXTRACTOR/apis/embedding_api.py)")



    # Start Uvicorn to serve FASTAPI app
    # Host '0.0.0.0' rende l'API accessibile dall'esterno (se sei su un server)
    # Porta 8000 è la porta standard
    uvicorn.run(app, host="localhost", port=8002)
