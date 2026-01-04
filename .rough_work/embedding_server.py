# src/rag_info_extractor/embedding_server.py


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# --- Config from env ---
MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "D:/Users/yye7607/.hf_models/embedding_models/e5-large-instruct") 
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

# # --- Load model once at startup ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Loading model from {MODEL_PATH} on device {device} ...")
# try:
#     model = SentenceTransformer(MODEL_PATH, device=device)
# except Exception as e:
#     raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

app = FastAPI(title="Local Embedding Server", version="1.0")


# Accept different request shapes to be compatible with other inference APIs
class EmbedRequest(BaseModel, Embeddings):
    # Many clients send 'input', or 'texts', or 'documents'
    model_name: str = Field(default=MODEL_PATH, alias="model")
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = False
    """Whether to show a progress bar."""
    
    # optional per-request batch_size override
    # batch_size: Optional[int] = None

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        # --- Load model once at startup ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {MODEL_PATH} on device {device} ...")
        try:
            self.model = SentenceTransformer(MODEL_PATH, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

    def _to_list(self, x: Union[str, List[str], None]) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i) for i in x]
        return [str(x)]


    def _embed(
        self,
        texts: List[str],
        encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed a text using the HuggingFace transformer model."""

        if not texts:
            raise HTTPException(status_code=400, detail="No input texts provided (use 'input', 'texts' or 'documents').")

        batch_size = self.encode_kwargs.get("batch_size", BATCH_SIZE) or self.query_encode_kwargs.get("batch_size", BATCH_SIZE) or BATCH_SIZE
        embeddings = []

        # encode in batches; sentence-transformers returns numpy arrays with convert_to_numpy=True
        texts = [x.replace("\n", " ") for x in texts]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                embs = self.model.encode(batch, **encode_kwargs)
            # ensure 2D
            if embs.ndim == 1:
                embs = np.expand_dims(embs, 0)
            embeddings.append(embs)

        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)
        
        embeddings = np.vstack(embeddings)

        return embeddings.tolist()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )
        return self._embed([text], embed_kwargs)[0]


    





# # Accept different request shapes to be compatible with other inference APIs
# class EmbedRequest(BaseModel):
#     # Many clients send 'input', or 'texts', or 'documents'
#     model: Optional[str] = None  # ignored, just for compatibility
#     input: Optional[Union[str, List[str]]] = None
#     texts: Optional[Union[str, List[str]]] = None
#     documents: Optional[Union[str, List[str]]] = None
#     # optional per-request batch_size override
#     batch_size: Optional[int] = None


# def _to_list(x: Union[str, List[str], None]) -> List[str]:
#     if x is None:
#         return []
#     if isinstance(x, list):
#         return [str(i) for i in x]
#     return [str(x)]


# @app.post("/embed")
# def embed(
#     req: EmbedRequest,
#     normalize_embeddings: bool = True
# ):
#     # combine fields in order of preference
#     texts = _to_list(req.input) or _to_list(req.texts) or _to_list(req.documents)
#     if not texts:
#         raise HTTPException(status_code=400, detail="No input texts provided (use 'input', 'texts' or 'documents').")

#     batch_size = req.batch_size or BATCH_SIZE
#     embeddings = []

#     # encode in batches; sentence-transformers returns numpy arrays with convert_to_numpy=True
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i : i + batch_size]
#         with torch.no_grad():
#             embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings = normalize_embeddings)
#         # ensure 2D
#         if embs.ndim == 1:
#             embs = np.expand_dims(embs, 0)
#         embeddings.append(embs)

#     embeddings = np.vstack(embeddings)
#     return {"embeddings": embeddings.tolist(), "shape": embeddings.shape}


# @app.get("/health")
# def health():
#     return {"status": "ok", "device": device, "model_path": MODEL_PATH}
