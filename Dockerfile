# Use the official NVIDIA CUDA 13.2 runtime for Ubuntu 24.04 (JetPack 7.2 compatible)
FROM nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# JetPack 7.2 utilizes Ubuntu 24.04, which defaults to Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrading setuptools and wheel only (skipping pip to avoid Debian conflict)
RUN python -m pip install --upgrade setuptools wheel --break-system-packages

# Pull PyTorch and ONNX Runtime directly from the Jetson AI Lab registry
RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp7/cu132 \
    torch torchvision onnxruntime-gpu --break-system-packages

COPY requirements.txt /app/requirements.txt
RUN grep -vE '^(torch|torchvision|torchaudio|onnxruntime|onnxruntime-gpu|numpy)([<=> ]|$)' /app/requirements.txt > /tmp/requirements_filtered.txt && \
    pip install -r /tmp/requirements_filtered.txt --break-system-packages

COPY . /app
RUN chmod +x /app/scripts/*.sh || true

RUN python -m py_compile /app/src/mtcaic4/infer.py && \
    python -m py_compile /app/src/mtcaic4/distill.py && \
    python -m mtcaic4.infer --help >/dev/null && \
    python -m mtcaic4.distill --help >/dev/null

CMD ["python", "-m", "mtcaic4.infer", "--help"]