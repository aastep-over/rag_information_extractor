import os
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
from transformers import AutoModel
from dotenv import load_dotenv
import yaml
import argparse
import torch
from pathlib import Path

# Logging
import logging
logger = logging.getLogger(__name__)
from rag_info_extractor.utils.common_logging import configure_logging
from rag_info_extractor.utils.load_config import cfgs

# --- Global Config ---
cfgs = cfgs.get("args", {})
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env.txt")

# Path to Pruner Model Directory/ name of the model
PRUNER_MODEL_NAME = cfgs.get("PRUNER_MODEL", "")
PRUNER_MODEL_NAME_ENV = PRUNER_MODEL_NAME.replace("/", "__").replace("-", "_").upper()
MODEL_NAME_OR_PATH = os.environ.get(PRUNER_MODEL_NAME_ENV, PRUNER_MODEL_NAME) # Load model from local path if already downloaded
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Input Data Structure ---
class PruneRequest(BaseModel):
    """Schema per la richiesta di pruning."""
    query: str = Field(..., description="La query originale dell'utente.")
    documents: List[str] = Field(..., description="Lista dei documenti da riordinare.")
    top_k: int = Field(5, description="specifica il numero di passaggi meglio classificati da mantenere per ogni domanda.")
    threshold: float = Field(0.05, description="quale soglia utilizzare per il pruning del contesto.")

class PruneResponse(BaseModel):
    """Schema per la risposta di pruning."""
    pruned_docs: List[str] = Field(..., description="Lista dei pruned (tagliare + rerank) documenti")

# --- Initialize server and model ---

pruner_model = None

# @app.on_event("startup")
@asynccontextmanager
async def load_model(app: FastAPI):
    """
    Funzione eseguita all'avvio dell'applicazione.
    Carica il modello in memoria RAM/VRAM una sola volta.
    """
    global pruner_model
    try:
        print(f"🔄 Caricamento del modello Pruner: {MODEL_NAME_OR_PATH}")
        # Load the Cross-Encoder model from path
        pruner_model = AutoModel.from_pretrained(
            MODEL_NAME_OR_PATH,
            trust_remote_code=True,
            local_files_only=True,
            device_map = DEVICE
        )
        logger.info(f"🖥️  Using device: {DEVICE.upper()}")
        logger.info("✅ Modello caricato con successo.")
    except Exception as e:
        logger.error(f"❌ Errore durante il caricamento del modello: {e}")
        raise HTTPException(status_code=500, detail=f"Impossibile caricare il modello: {e}")
    
    yield

    # Clear value of pruner
    pruner_model = None


app = FastAPI(
    title="RAG Pruner API",
    description="Servizio dedicato al caricamento e all'inferenza del modello xprovence-reranker-bgem3-v1.",
    lifespan=load_model
)

# --- Endpoint API ---
@app.post(
    "/prune", 
    response_model=PruneResponse, 
    summary="Esegue il pruning di una lista di documenti."
)
def prune_documents(request: PruneRequest):
    """
    Prende una query e una lista di documenti, e restituisce i punteggi di pertinenza.
    """
    if pruner_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Il modello non è ancora stato caricato o non è disponibile."
        )
    
    # 1. Esegui l'inferenza
    # Il metodo predict è thread-safe per Sentence-Transformers
    try:
        pruned = pruner_model.process(
            [request.query],
            [request.documents],
            batch_size=32 if torch.cuda.is_available() else 4,
            reorder=True,
            top_k=request.top_k,
            threshold=request.threshold
        )
    except Exception as e:
        # Gestione di errori di inferenza (es. batch size troppo grande)
        raise HTTPException(
            status_code=500, 
            detail=f"Errore durante l'inferenza del modello: {e}"
        )

    # 3. Restituisci i risultati
    return PruneResponse(pruned_docs=[c for c in pruned['pruned_context'][0]])




# --- Start Server ---
if __name__ == "__main__":
    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\re_ranker_api.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Launching Pruner API (RAG_CHATBOT/apis/pruner_api.py)")

    PORT = int(os.getenv("PORT", 8001))

    # Start Uvicorn to serve FASTAPI app
    uvicorn.run(app, host="localhost", port=PORT)
