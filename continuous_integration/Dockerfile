ARG BASE_IMAGE=rapidsai/devcontainers:23.08-cpp-cuda11.8-mambaforge-ubuntu22.04
FROM ${BASE_IMAGE} as stage0

ARG USE_CUDA=1
ENV USE_CUDA=${USE_CUDA}

ARG BUILD_MARCH=nocona
ENV BUILD_MARCH=${BUILD_MARCH}

ARG AWS_REGION
ENV AWS_REGION=${AWS_REGION}
ENV SCCACHE_REGION=${AWS_REGION:-${SCCACHE_REGION}}

ENV DEFAULT_CONDA_ENV=legate
ENV PYTHONDONTWRITEBYTECODE=1
ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev
ENV VAULT_HOST=https://vault.ops.k8s.rapids.ai
# 8 hours
ENV VAULT_S3_TTL="28800s"

SHELL ["/bin/bash", "-Eeox", "pipefail", "-c"]

USER coder

WORKDIR /home/coder/.artifacts
WORKDIR /home/coder

COPY --chown=coder:coder continuous_integration/home/coder/.local/bin/* /home/coder/.local/bin/
COPY --chown=coder:coder . /home/coder/legate

#---------------------------------------------------
FROM stage0 as setup

RUN make-conda-env

#---------------------------------------------------
FROM setup as build

# Mount in secrets
RUN --mount=type=secret,id=GH_TOKEN,uid=1000,gid=1000              \
    --mount=type=secret,id=AWS_SESSION_TOKEN,uid=1000,gid=1000     \
    --mount=type=secret,id=AWS_ACCESS_KEY_ID,uid=1000,gid=1000     \
    --mount=type=secret,id=AWS_SECRET_ACCESS_KEY,uid=1000,gid=1000 \
<<EOF

{ set +xo history; } 2>/dev/null;
eval "export $(find /run/secrets/ -type f -exec bash -c 'echo ${0/\/run\/secrets\//}=$(<${0})' {} \;)";
{ set -xo history; } 2>/dev/null;

if ! test -n "${GH_TOKEN:-}" \
&& ! test -n "${AWS_ACCESS_KEY_ID:-}" \
&& ! test -n "${AWS_SECRET_ACCESS_KEY:-}" ; then
    export SCCACHE_BUCKET="";
fi

if test -n "${GH_TOKEN:-}"; then echo "Has GH_TOKEN"; fi
if test -n "${AWS_ACCESS_KEY_ID:-}"; then echo "Has AWS_ACCESS_KEY_ID"; fi
if test -n "${AWS_SECRET_ACCESS_KEY:-}"; then echo "Has AWS_SECRET_ACCESS_KEY"; fi
if test -n "${SCCACHE_BUCKET:-}"; then echo "Has SCCACHE_BUCKET"; fi

. devcontainer-utils-post-attach-command;

build-all;

EOF

#---------------------------------------------------
FROM stage0 as final

COPY --from=build /home/coder/.artifacts /home/coder/.artifacts

# Create a fresh conda env from the new legate_core package

RUN mamba create -yn ${DEFAULT_CONDA_ENV:-legate}   \
 && mamba install -yn ${DEFAULT_CONDA_ENV:-legate}  \
    # local legate_core channel first
    -c ~/.artifacts/legate_core                     \
    # then conda-forge
    -c conda-forge                                  \
    # and finally nvidia
    -c nvidia                                       \
    legate-core                                     \
    mypy pytest pytest-mock ipython jupyter_client  \
    ;
