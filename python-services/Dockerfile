# syntax=docker/dockerfile:1.7

# CUDA 12.4 matches the pinned torch/torchaudio wheels in requirements.txt.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        libvulkan1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PY_BIND_HOST=0.0.0.0 \
    HF_HOME=/models/huggingface

RUN mkdir -p /models/huggingface /work

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=8s --start-period=180s --retries=20 \
  CMD python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)); assert d.get('status') == 'ok'"

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
