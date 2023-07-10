#! /usr/bin/env bash

# cd to the repo root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

exec ./scripts/launch-devcontainer-conda.sh bash -licx '
# Create build env + build libs and conda package
make-conda-env;
mamba run -n "${DEFAULT_CONDA_ENV:-legate}" --cwd ~/ --live-stream build-legate-cpp;
mamba run -n "${DEFAULT_CONDA_ENV:-legate}" --cwd ~/ --live-stream build-legate-wheel;
mamba run -n "${DEFAULT_CONDA_ENV:-legate}" --cwd ~/ --live-stream build-legate-conda;

# Uncreate build env
mv ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}{,.bak};

# Recreate env from conda package
mamba create -y -n "${DEFAULT_CONDA_ENV:-legate}"        \
    -c /tmp/out/legate_core -c conda-forge -c nvidia     \
    legate-core pytest pytest-mock ipython jupyter_client;

# Test package
mamba run -n legate --cwd ~/legate/tests/unit --live-stream pytest;
';
