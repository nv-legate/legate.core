#! /usr/bin/env bash

# cd to the repo root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

exec ./scripts/launch-devcontainer-conda.sh bash -licx '
# Create build env
# make-conda-env;
bash -c ". conda-utils; get_yaml_and_make_conda_env";

# Activate build env
. /etc/profile.d/*-mambaforge.sh;

# Build libs + conda package
build-legate-all;

# Uncreate build env
rm -rf ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}.bak;
mv ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}{,.bak};

# Recreate env from conda package
mamba create -y -n "${DEFAULT_CONDA_ENV:-legate}"        \
    `# local legate_core channel first`                  \
    -c /tmp/conda-build/legate_core                      \
    `# then conda-forge`                                 \
    -c conda-forge                                       \
    `# and finally nvidia`                               \
    -c nvidia                                            \
    legate-core                                          \
    mypy pytest pytest-mock ipython jupyter_client       ;

# Check types
run-test-or-analysis mypy;

# Test package
run-test-or-analysis unit;

# Restore build env
rm -rf ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}/;
mv ~/.conda/envs/${DEFAULT_CONDA_ENV:-legate}{.bak,};
';
