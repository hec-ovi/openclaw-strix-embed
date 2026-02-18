#!/bin/bash
set -e

# =============================================================================
# OpenClaw Embeddings — Entrypoint
# =============================================================================
# 1. Check GPU availability
# 2. Start uvicorn (model downloads on first request if not cached)

MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"

log() {
    echo "[embeddings] $1"
}

log "========================================"
log "Model: $MODEL"
log "========================================"

if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
    export HSA_OVERRIDE_GFX_VERSION=11.5.1
fi

log "Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'[embeddings] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[embeddings] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'[embeddings] ROCm/HIP: {torch.version.hip}')
    print(f'[embeddings] Device: cuda')
else:
    print('[embeddings] WARNING: No GPU detected, falling back to CPU')
    print('[embeddings] Device: cpu')
"

log "Starting server on port 80..."
exec uvicorn server:app --host 0.0.0.0 --port 80 --workers 1
