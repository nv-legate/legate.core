#! /usr/bin/env bash

cd $(dirname "$(realpath "$0")")/..

# Use sccache if installed
source ./scripts/util/build-caching.sh
# Use consistent C[XX]FLAGS
source ./scripts/util/compiler-flags.sh
# Uninstall existing globally-installed Legion and legate_core (if installed)
source ./scripts/util/uninstall-global-legion-and-legate-core.sh

# Remove existing build artifacts
rm -rf ./{build,_skbuild,dist,legate_core.egg-info}

# Define CMake configuration arguments
cmake_args="${CMAKE_ARGS:-}"

# Use ninja-build if installed
if [[ -n "$(which ninja)" ]]; then cmake_args+=" -GNinja"; fi

# Add other build options here as desired
cmake_args+="
-D Legion_USE_CUDA=ON
-D Legion_USE_OpenMP=ON
-D Legion_CUDA_ARCH=native
";

# Use all but 2 threads to compile
ninja_args="-j$(nproc --ignore=2)"

# Build legion_core + legion_core_python and install into the current Python environment
SKBUILD_BUILD_OPTIONS="$ninja_args"       \
CMAKE_ARGS="$cmake_args"                  \
    python -m pip install                 \
        --root / --prefix "$CONDA_PREFIX" \
        --no-deps --no-build-isolation    \
        --upgrade                         \
        . -vv

# Install Legion's Python CFFI bindings
cmake \
    --install _skbuild/*/cmake-build/_deps/legion-build/bindings/python \
    --prefix "$CONDA_PREFIX"
