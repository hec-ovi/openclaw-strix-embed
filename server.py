"""OpenAI-compatible Embeddings API server."""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
CACHE_DIR = os.environ.get("MODEL_CACHE", "/data/models")

app = FastAPI(
    title="OpenClaw Embeddings",
    description=(
        "Local, GPU-accelerated, OpenAI-compatible embeddings API. "
        "Drop-in replacement for OpenAI `/v1/embeddings` — no API keys, "
        "no usage fees, no data leaving your network."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

model: SentenceTransformer = None


@app.on_event("startup")
def load_model():
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embeddings] Loading {MODEL_ID} on {device}")
    if torch.cuda.is_available():
        print(f"[embeddings] GPU: {torch.cuda.get_device_name(0)}")
    model = SentenceTransformer(MODEL_ID, cache_folder=CACHE_DIR, device=device)
    print(f"[embeddings] Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")


# ── Request / Response models ────────────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    model: str = Field(default=MODEL_ID, description="Model ID to use for embedding.", examples=["BAAI/bge-m3"])
    input: str | list[str] = Field(description="Text or array of texts to embed.", examples=["Hello world"])
    encoding_format: str = Field(default="float", description="Encoding format. Only `float` is supported.")


class EmbeddingData(BaseModel):
    object: str = Field(default="embedding", description="Object type, always `embedding`.")
    index: int = Field(description="Index of this embedding in the input array.")
    embedding: list[float] = Field(description="The embedding vector as an array of floats.")


class UsageInfo(BaseModel):
    prompt_tokens: int = Field(description="Approximate number of tokens in the input.")
    total_tokens: int = Field(description="Same as prompt_tokens (no completion tokens for embeddings).")


class EmbeddingResponse(BaseModel):
    object: str = Field(default="list", description="Object type, always `list`.")
    data: list[EmbeddingData] = Field(description="Array of embedding objects.")
    model: str = Field(description="Model ID used.")
    usage: UsageInfo = Field(description="Token usage statistics.")

    model_config = {"json_schema_extra": {"examples": [{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, -0.044, 0.0078]}], "model": "BAAI/bge-m3", "usage": {"prompt_tokens": 2, "total_tokens": 2}}]}}


class ModelInfo(BaseModel):
    id: str = Field(description="Model ID.")
    object: str = Field(default="model", description="Object type, always `model`.")
    owned_by: str = Field(default="local", description="Owner.")
    meta: dict = Field(description="Model metadata including dimension.")


class ModelListResponse(BaseModel):
    object: str = Field(default="list", description="Object type, always `list`.")
    data: list[ModelInfo] = Field(description="Array of available models.")


class HealthResponse(BaseModel):
    status: str = Field(description="Service status.")
    model_loaded: bool = Field(description="Whether the embedding model is loaded and ready.")
    model: str = Field(description="Model ID.")
    device: str = Field(description="Compute device (`cuda` or `cpu`).")
    dimension: int = Field(description="Embedding vector dimension.")


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings",
    description="Generate embedding vectors for the given input text(s). OpenAI-compatible.",
    tags=["Embeddings"],
)
def create_embeddings(req: EmbeddingRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    texts = [req.input] if isinstance(req.input, str) else req.input

    embeddings = model.encode(texts, normalize_embeddings=True)

    data = []
    total_tokens = 0
    for i, emb in enumerate(embeddings):
        data.append(EmbeddingData(
            index=i,
            embedding=emb.tolist(),
        ))
        total_tokens += len(texts[i].split())

    return EmbeddingResponse(
        data=data,
        model=MODEL_ID,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


@app.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List models",
    description="List available embedding models and their metadata.",
    tags=["Models"],
)
def list_models():
    dim = model.get_sentence_embedding_dimension() if model else 0
    return ModelListResponse(
        data=[ModelInfo(
            id=MODEL_ID,
            meta={"dimension": dim},
        )],
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status, model readiness, device info, and embedding dimension.",
    tags=["System"],
)
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = model.get_sentence_embedding_dimension() if model else 0
    return HealthResponse(
        status="ok" if model else "loading",
        model_loaded=model is not None,
        model=MODEL_ID,
        device=device,
        dimension=dim,
    )
