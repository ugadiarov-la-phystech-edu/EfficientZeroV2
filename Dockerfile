FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    python3.8-dev \
    git \
    g++ \
    libgl1-mesa-glx \
    libstdc++6 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

RUN ln -s /usr/bin/python3.8 /usr/bin/python