# =============================================================================
# OpenClaw Embeddings — Local OpenAI-compatible Embeddings API
# =============================================================================
# Base: Ubuntu Rolling + ROCm PyTorch (gfx1151 / Strix Halo)
# Features: FastAPI server with /v1/embeddings endpoint, GPU-accelerated

FROM ubuntu:rolling

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_CACHE_DIR=/root/.cache/uv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgl1 libglib2.0-0 libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# PyTorch ROCm gfx1151 from AMD prerelease index + embedding dependencies
RUN uv venv .venv --python 3.12 && \
    uv pip install --pre \
    torch torchvision \
    --index-url https://rocm.prereleases.amd.com/whl/gfx1151/ && \
    uv pip install \
    sentence-transformers \
    fastapi \
    uvicorn[standard]

COPY server.py /app/server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
