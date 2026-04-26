<h1 align="center">openclaw-strix-embed</h1>

<p align="center">
  <strong>Local, GPU-accelerated, OpenAI-compatible <code>/v1/embeddings</code> API on AMD Strix Halo. BAAI/bge-m3 by default, no API keys, no fees, no data leaving your network.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/AMD-Strix_Halo-ED1C24?logo=amd&logoColor=white" alt="AMD Strix Halo" />
  <img src="https://img.shields.io/badge/ROCm-7.x-EF5B25?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/Model-BAAI/bge--m3-7B3FA0" alt="Model" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

---

## What this is

Local, GPU-accelerated, OpenAI-compatible embeddings API. Drop-in replacement for OpenAI's `/v1/embeddings` endpoint, no API keys, no usage fees, no data leaving your network.

Built for AMD Strix Halo (RDNA 3.5 / gfx1151) with ROCm, but falls back to CPU if no GPU is available.

## Why

Every vector database and RAG pipeline needs an embeddings API. The standard options, OpenAI `text-embedding-3-small`, Google `text-embedding-004`, charge per token and send your data to external servers. This runs the same API contract locally, for free, on your own hardware.

## Project Structure

```
.
├── .gitignore              # Ignores .env and data/
├── .env.template           # Template, copy to .env
├── README.md               # This file
├── llm.txt                 # Complete technical reference
├── Dockerfile              # Ubuntu Rolling + ROCm PyTorch + FastAPI
├── docker-compose.yml      # Service definition with GPU passthrough
├── entrypoint.sh           # GPU check + uvicorn start
├── server.py               # OpenAI-compatible FastAPI server
└── data/                   # Persistent model cache (git-ignored)
    └── models/             # HuggingFace model files
```

## Prerequisites

- Docker with compose plugin
- AMD Strix Halo (or any RDNA 3.5 GPU) for GPU mode
- ~7 GB disk for the model + Docker image

## Quick Start

```bash
cp .env.template .env
# Edit .env if you want to change the model or port
docker compose up -d --build
```

First start downloads the model (~4.3 GB). Subsequent starts load from cache in seconds.

**Verify:**
```bash
# Check GPU detection
docker logs openclaw-embeddings

# Test the API
curl http://localhost:8484/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":"Hello world"}'
```

## API

### POST /v1/embeddings

OpenAI-compatible. Works with any client that speaks the OpenAI embeddings format.

**Request:**
```json
{
  "model": "BAAI/bge-m3",
  "input": "text to embed"
}
```

`input` accepts a single string or an array of strings for batch embedding.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.044, ...]
    }
  ],
  "model": "BAAI/bge-m3",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

### GET /v1/models

Lists available models and their dimensions.

### GET /health

Returns `{"status": "ok", "model_loaded": true}` when ready.

## Configuration

All configurable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model ID |
| `EMBEDDING_PORT` | `8484` | Host port for the API |

## Model

Default: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

| Property | Value |
|----------|-------|
| Dimensions | 1024 |
| Languages | 100+ (excellent EN + ES) |
| Max tokens | 8192 |
| Size | ~2.2 GB (weights) |

You can swap it for any [sentence-transformers](https://www.sbert.net/) compatible model by changing `EMBEDDING_MODEL` in `.env`.

## Using with Multipass VMs

If OpenClaw runs inside a Multipass VM and this embeddings service runs on the host, `localhost` won't work, it points to the VM, not the host.

**Find the host IP on the Multipass bridge:**
```bash
# On the host
ip addr show mpqemubr0 | grep 'inet '
# Example output: inet 10.5.162.1/24 ...
```

**Test from inside the VM:**
```bash
multipass exec <vm-name> -- curl http://10.5.162.1:8484/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":"connectivity test"}'
```

**Configure OpenClaw with:**

| Setting | Value |
|---------|-------|
| Base URL | `http://<host-bridge-ip>:8484/v1` |
| Model | `BAAI/bge-m3` |
| Auth | None |

## Performance

Tested on AMD Ryzen AI Max (Strix Halo) with Radeon 8060S iGPU:

| Metric | Value |
|--------|-------|
| Latency (single text) | ~47ms |
| Latency (first request, cold) | ~270ms |
| GPU VRAM used | ~1.5 GB |
| Model load time (from cache) | ~10s |

## GPU Details

This container uses the same ROCm setup from [rocm-strix-docker](https://github.com/hec-ovi/rocm-strix-docker):

- **`HSA_OVERRIDE_GFX_VERSION=11.5.1`**, required for ROCm to recognize Strix Halo
- **`privileged: true`**, grants `/dev/kfd` and `/dev/dri` access for GPU compute
- **`ipc: host`**, shared memory for PyTorch
- **PyTorch wheels** from `https://rocm.prereleases.amd.com/whl/gfx1151/` (ROCm 7.11 prerelease)
- **UV** manages Python 3.12 + all packages (no pip)

---

## License

[MIT](LICENSE) for original code in this repository (FastAPI server, Dockerfile, Compose configs, scripts). Third-party model weights (BAAI/bge-m3) and runtimes (sentence-transformers, transformers, PyTorch ROCm) retain their own upstream licenses; this repository does not redistribute them.

## Verified Output

```
[embeddings] ========================================
[embeddings] Model: BAAI/bge-m3
[embeddings] ========================================
[embeddings] Checking GPU...
[embeddings] GPU: Radeon 8060S Graphics
[embeddings] VRAM: 124.6 GB
[embeddings] ROCm/HIP: 7.2.53150-7b886380f9
[embeddings] Device: cuda
[embeddings] Starting server on port 80...
[embeddings] Loading BAAI/bge-m3 on cuda
[embeddings] GPU: Radeon 8060S Graphics
[embeddings] Model loaded. Dimension: 1024
```
