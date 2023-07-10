#! /usr/bin/env bash

# cd to the repo root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

exec ./scripts/launch-devcontainer.sh \
    rapidsai/devcontainers:23.08-cpp-rust-cuda11.8-ubuntu22.04 \
    $(if ! test -z "${@}"; then echo "${@}"; else echo "bash -li"; fi);
