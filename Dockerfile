FROM python:3.12-slim

# ffmpeg нужен для голосовых/audio (faster-whisper умеет webm/ogg через ffmpeg)
# libgomp1 — рантайм OpenMP для ctranslate2 (внутри faster-whisper)
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
    HF_HOME=/app/data/cache/huggingface

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY main.py healthcheck.py ./
COPY alembic.ini .
COPY alembic/ alembic/

# data — монтируется томом снаружи (БД, сессии, qdrant, media, кэш моделей)
RUN mkdir -p /app/data \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python healthcheck.py || exit 1

CMD ["python", "main.py"]
