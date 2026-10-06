FROM python:3.12-slim

# libglib/libgomp: exigidas pelo onnxruntime e pelo OpenCV mesmo na versão headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# dependências antes do código: o cache desta camada só se perde quando pyproject/uv.lock mudam
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY scripts ./scripts

# os modelos (DINOv2, SigLIP2 ≈ 1,5 GB) e os índices ficam em volumes: não entram na imagem
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
