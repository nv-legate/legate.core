FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as llr_base

RUN apt-get update && apt-get -y install cmake git rust-all wget curl

ENV PATH="/root/miniconda3/bin:$PATH"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -bf && \
    conda init bash

COPY scripts /build/legate.core/scripts

WORKDIR /build/legate.core

RUN scripts/generate-conda-envs.py --python 3.10 --ctk 11.8 --os linux --compilers --openmpi

RUN conda env create -n legate -f environment-test-linux-py3.10-cuda11.8-compilers-openmpi.yaml

SHELL ["/bin/bash", "--login", "-c"]

# Make sure pynvml is installed
RUN conda run -n legate /bin/bash -c "python -c \"import pynvml\""


