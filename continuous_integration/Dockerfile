ARG BASE_IMAGE=rapidsai/devcontainers:23.06-cpp-cuda11.8-mambaforge-ubuntu22.04
FROM ${BASE_IMAGE} as stage0

ENV PYTHONDONTWRITEBYTECODE=1
ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-east"
ENV SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev
ENV VAULT_HOST=https://vault.ops.k8s.rapids.ai
ENV HISTFILE=/home/coder/.cache/._bash_history
ENV DEFAULT_CONDA_ENV=legate

ARG USE_CUDA
ENV USE_CUDA=${USE_CUDA}

ENV BUILD_MARCH=nocona

ENV PATH="${PATH}:/home/coder/.local/bin"

USER coder

WORKDIR /home/coder/.cache
WORKDIR /home/coder/.artifacts
WORKDIR /home/coder

COPY --chown=coder:coder continuous_integration/home/coder/.gitconfig /home/coder/
COPY --chown=coder:coder continuous_integration/home/coder/.local/bin/* /home/coder/.local/bin/
COPY --chown=coder:coder . /home/coder/legate

RUN chmod a+x /home/coder/.local/bin/* && \
    mkdir -p /tmp/out && \
    chown -R coder:coder /tmp/out  && \
    chown -R coder:coder /home/coder/.artifacts

#---------------------------------------------------
FROM stage0 as setup

RUN get-yaml-and-make-conda-env

#---------------------------------------------------
FROM setup as build
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}
ARG AWS_SESSION_TOKEN
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# If .creds exists copy it to /run/secrets
COPY --chown=coder:coder .cred[s] /run/secrets

RUN entrypoint build-all

#---------------------------------------------------
FROM stage0 as final
COPY --from=build /tmp/out /tmp/out
COPY --from=build /tmp/conda-build /tmp/conda-build
COPY --from=build /tmp/env_yaml /tmp/env_yaml
