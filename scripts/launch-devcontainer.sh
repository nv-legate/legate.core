#! /usr/bin/env bash

launch_devcontainer() {

    set -Eeuox pipefail;

    # cd to the repo root
    cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

    local cwd="$(pwd)";

    local args=();
    args+=(--rm);
    args+=(--tty);
    args+=(--interactive);
    args+=(--gpus all);
    args+=(--user coder);
    args+=(--workdir /home/coder/legate);
    args+=(--entrypoint devcontainer-utils-post-attach-command-entrypoint);

    local vars=();
    vars+=(-e "PYTHONSAFEPATH=1");
    vars+=(-e "DEFAULT_CONDA_ENV=legate");
    vars+=(-e "PYTHONDONTWRITEBYTECODE=1");

    vars+=(-e "HISTFILE=/home/coder/.local/state/._bash_history");

    vars+=(-e "SCCACHE_REGION=us-east-2");
    vars+=(-e "SCCACHE_BUCKET=rapids-sccache-devs");
    vars+=(-e "SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev");
    vars+=(-e "VAULT_HOST=https://vault.ops.k8s.rapids.ai");

    local vols=();
    vols+=(-v "${cwd}:/home/coder/legate");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/build-legate-all:/home/coder/.local/bin/build-legate-all");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/build-legate-conda:/home/coder/.local/bin/build-legate-conda");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/build-legate-cpp:/home/coder/.local/bin/build-legate-cpp");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/build-legate-wheel:/home/coder/.local/bin/build-legate-wheel");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/conda-utils:/home/coder/.local/bin/conda-utils");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/copy-artifacts:/home/coder/.local/bin/copy-artifacts");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/entrypoint:/home/coder/.local/bin/entrypoint");
    vols+=(-v "${cwd}/continuous_integration/home/coder/.local/bin/run-test-or-analysis:/home/coder/.local/bin/run-test-or-analysis");

    for x in ".aws" ".cache" ".cargo" ".config" ".conda/pkgs" ".local/state" "legion"; do
        if test -d "${cwd}/../${x}"; then
            vols+=(-v "${cwd}/../${x}:/home/coder/${x}");
        fi
    done

    if test -n "${SSH_AUTH_SOCK:-}"; then
        vars+=(-e "SSH_AUTH_SOCK=/tmp/ssh-auth-sock");
        vols+=(-v "${SSH_AUTH_SOCK}:/tmp/ssh-auth-sock");
    fi

    exec docker run ${args[@]} ${vars[@]} ${vols[@]} "${@}";
}

launch_devcontainer "$@";
