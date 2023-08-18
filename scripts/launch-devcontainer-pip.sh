#! /usr/bin/env bash

# cd to the repo root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

if test ${#@} -le 0; then set -- bash -li; fi

exec ./scripts/launch-devcontainer.sh \
    rapidsai/devcontainers:23.10-cpp-rust-cuda11.8-ubuntu22.04 \
    "${@}";
