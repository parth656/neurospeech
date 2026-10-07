FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    WHISPER_MODEL_SIZE=base.en \
    WHISPER_CACHE_DIR=/app/whisper_models \
    HF_HOME=/tmp/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Pre-download Whisper at build time so the first user doesn't wait / time out
RUN python -c "import os, whisper; whisper.load_model(os.environ['WHISPER_MODEL_SIZE'], device='cpu', download_root=os.environ['WHISPER_CACHE_DIR'])" && \
    chmod -R a+rX /app/whisper_models

COPY src/ ./src/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8501/_stcore/health || exit 1

# XSRF/CORS off: avoids 403 errors on mic recording and file upload inside the HF Spaces iframe
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.enableXsrfProtection=false", "--server.enableCORS=false"]