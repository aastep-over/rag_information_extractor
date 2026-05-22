import argparse
# Logging
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.load_config import cfgs

# --- Global Config ---
cfgs = cfgs.get("args", {})
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env.txt")

# Path to ReRanker Model Directory/ name of the model
RERANKER_MODEL_NAME = cfgs.get("RERANKER_MODEL", "")
RERANKER_MODEL_NAME_ENV = (
    RERANKER_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
)
MODEL_NAME_OR_PATH = os.environ.get(
    RERANKER_MODEL_NAME_ENV, RERANKER_MODEL_NAME
)  # Load model from local path if already downloaded
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Input Data Structure ---
class RerankRequest(BaseModel):
    """Schema per la richiesta di re-ranking."""

    query: str = Field(..., description="La query originale dell'utente.")
    documents: List[str] = Field(..., description="Lista dei documenti da riordinare.")


class RerankResponse(BaseModel):
    """Schema per la risposta di re-ranking."""

    scores: List[float] = Field(
        ...,
        description="Lista dei punteggi (score) di pertinenza, corrispondenti ai documenti.",
    )


# --- Initialize server and model ---

reranker_model = None


# @app.on_event("startup")
@asynccontextmanager
async def load_model(app: FastAPI):
    """
    Funzione eseguita all'avvio dell'applicazione.
    Carica il modello in memoria RAM/VRAM una sola volta.
    """
    global reranker_model
    try:
        logger.info(f"🔄 Caricamento del modello Re-ranker: {MODEL_NAME_OR_PATH}")
        # Load the Cross-Encoder model from path
        reranker_model = CrossEncoder(MODEL_NAME_OR_PATH, device=DEVICE, max_length=512)
        logger.info(f"🖥️  Using device: {DEVICE.upper()}")
        logger.info("✅ Modello caricato con successo.")
    except Exception as e:
        logger.error(f"❌ Errore durante il caricamento del modello: {e}")
        raise HTTPException(
            status_code=500, detail=f"Impossibile caricare il modello: {e}"
        )

    yield

    # Clear value of reranker
    reranker_model = None


app = FastAPI(
    title="RAG Reranker API",
    description="Servizio dedicato al caricamento e all'inferenza del modello Cross-Encoder.",
    lifespan=load_model,
)


# --- Endpoint API ---
@app.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Esegue il re-ranking di una lista di documenti.",
)
def rerank_documents(request: RerankRequest):
    """
    Prende una query e una lista di documenti, e restituisce i punteggi di pertinenza.
    """
    if reranker_model is None:
        raise HTTPException(
            status_code=503,
            detail="Il modello non è ancora stato caricato o non è disponibile.",
        )

    # 1. Prepara i dati di input per il modello (coppie di (query, document))
    # Il Cross-Encoder accetta una lista di liste/tuple
    sentence_pairs = [(request.query, doc) for doc in request.documents]

    # 2. Esegui l'inferenza
    # Il metodo predict è thread-safe per Sentence-Transformers
    try:
        scores = reranker_model.predict(
            sentence_pairs,
            batch_size=32 if torch.cuda.is_available() else 4,
            show_progress_bar=False,
        ).tolist()
    except Exception as e:
        # Gestione di errori di inferenza (es. batch size troppo grande)
        raise HTTPException(
            status_code=500, detail=f"Errore durante l'inferenza del modello: {e}"
        )

    # 3. Restituisci i risultati
    return RerankResponse(scores=scores)


# --- Start the Server ---
if __name__ == "__main__":
    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )  # For DEBUG level logging, run in cli: python .\re_ranker_api.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Launching ReRanker API (RAG_CHATBOT/apis/re_ranker_api.py)")

    # PORT = int(os.getenv("PORT", 8000))

    # Start Uvicorn to serve FASTAPI app
    uvicorn.run(app, host="localhost", port=8000)
