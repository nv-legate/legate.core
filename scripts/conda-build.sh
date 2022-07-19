#! /usr/bin/env bash

cd $(dirname "$(realpath "$0")")/..

mkdir -p /tmp/conda-build
rm -rf /tmp/conda-build/*
mkdir -p /tmp/conda-build/out

PYTHON_VERSION="${PYTHON_VERSION:-3.9}"

CUDA="$(nvcc --version | head -n4 | tail -n1 | cut -d' ' -f5 | cut -d',' -f1).*" \
conda mambabuild \
    --numpy 1.22 \
    --python $PYTHON_VERSION \
    --override-channels \
    -c conda-forge -c nvidia \
    --croot /tmp/conda-build \
    --prefix-length 3 \
    --no-test \
    --no-verify \
    --build-id-pat='' \
    --merge-build-host \
    --no-include-recipe \
    --no-anaconda-upload \
    --output-folder /tmp/conda-build/out \
    --variants "{gpu_enabled: 'true', python: $PYTHON_VERSION}" \
    ./conda/conda-build
