FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as llr_base

RUN apt-get update && apt-get -y install cmake git rust-all wget
WORKDIR /app

ENV PATH="/root/miniconda3/bin:$PATH"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -bf && \
    conda init bash

