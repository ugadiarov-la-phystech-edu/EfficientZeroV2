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

RUN ln -s /usr/bin/python3.8 /usr/bin/python

WORKDIR /app

#RUN git clone -b shapes2d_learn --single-branch https://github.com/ugadiarov-la-phystech-edu/EfficientZeroV2.git .

RUN python -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app/ez/mcts/ctree
RUN bash make.sh

WORKDIR /app/ez/mcts/ctree_v2
RUN bash make.sh

WORKDIR /app/ez/mcts/ori_ctree
RUN bash make.sh

WORKDIR /app