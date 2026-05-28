# ============================================================================
# Stage 1: Builder — install deps + pip packages
# ============================================================================
FROM python:3.13-slim AS builder

# Build dependencies (needed for ctranslate2, torch, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc g++ make \
       ffmpeg \
       libgomp1 \
       ca-certificates \
       tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python packages into /venv
COPY requirements.txt ./
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r requirements.txt

# ============================================================================
# Stage 2: Runner — minimal production image
# ============================================================================
FROM python:3.13-slim AS runner

# Runtime only — no build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libgomp1 \
       ca-certificates \
       tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/app/data/cache \
    HF_HOME=/app/data/cache/huggingface \
    PATH="/venv/bin:$PATH"

WORKDIR /app

# Copy virtual env from builder (all pip packages)
COPY --from=builder /venv /venv

# Copy application code
COPY src/ ./src/
COPY skills/ ./skills/
COPY docs/ ./docs/
COPY main.py healthcheck.py ./
COPY alembic.ini .
COPY alembic/ alembic/

# data — mounted as volume (DB, sessions, qdrant, media, model cache)
RUN mkdir -p /app/data \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python healthcheck.py || exit 1

CMD ["python", "main.py"]
