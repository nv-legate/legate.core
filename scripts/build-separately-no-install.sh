#! /usr/bin/env bash

cd $(dirname "$(realpath "$0")")/..

# Use sccache if installed
source ./scripts/util/build-caching.sh
# Use consistent C[XX]FLAGS
source ./scripts/util/compiler-flags.sh

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
-D Legion_USE_Python=ON
-D Legion_Python_Version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f3 --complement)
-D Legion_CUDA_ARCH=native
";

# Use all but 2 threads to compile
ninja_args="-j$(nproc --ignore=2)"

# Configure legate_core C++
cmake -S . -B build ${cmake_args}

# Build legate_core C++
cmake --build build ${ninja_args}

# Pretend to install Legion because Legion's CMakeLists only generates the Legion CFFI bindings at install time
(
    tmpdir=$(mktemp -d);
    cmake --install build/_deps/legion-build --prefix "$tmpdir" &>/dev/null;
    rm -rf "$tmpdir";
)

cmake_args+="
-D FIND_LEGATE_CORE_CPP=ON
-D legate_core_ROOT=$(pwd)/build
"

# Build legion_core_python and perform an "editable" install
SKBUILD_BUILD_OPTIONS="$ninja_args"       \
CMAKE_ARGS="$cmake_args"                  \
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
    python -m pip install                 \
        --root / --prefix "$CONDA_PREFIX" \
        --no-deps --no-build-isolation    \
        --editable                        \
        . -vv
