#! /usr/bin/env bash

# cd to the repo root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

exec ./scripts/launch-devcontainer-conda.sh bash -licx '
# Create build env
make-conda-env;

# Build libs + conda package
build-all;

# Uncreate build env
rm -rf ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}.bak;
mv ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}{,.bak};

# Recreate env from conda package
mamba create -y -n "${DEFAULT_CONDA_ENV:-legate}"        \
    -c ~/.artifacts/legate_core -c conda-forge -c nvidia \
    legate-core pytest pytest-mock ipython jupyter_client;

# Check types
type-check-legate-python;

# Test package
test-legate-python;

# Restore build env
rm -rf ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}/;
mv ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}{.bak,};
';
