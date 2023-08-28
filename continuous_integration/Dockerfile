ARG BASE_IMAGE
FROM ${BASE_IMAGE} as stage0

SHELL ["/bin/bash", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1
ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-east"
ENV SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev
ENV VAULT_HOST=https://vault.ops.k8s.rapids.ai
ENV HISTFILE=/home/coder/.cache/._bash_history
ENV DEFAULT_CONDA_ENV=legate

ARG USE_CUDA
ENV USE_CUDA=${USE_CUDA}

ENV USE_OPENMP=ON

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
    chown -R coder:coder /tmp/out

#---------------------------------------------------
FROM stage0 as setup

USER coder
WORKDIR /home/coder

RUN set -x && . conda-utils && get_yaml_and_make_conda_env

#---------------------------------------------------
FROM setup as build
USER coder
WORKDIR /home/coder

ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}
ARG AWS_SESSION_TOKEN
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

COPY --chown=coder:coder .creds /run/secrets

RUN entrypoint build-legate-all

#---------------------------------------------------
FROM stage0 as final
USER coder
WORKDIR /home/coder

COPY --from=build --chown=coder:coder /tmp/out /tmp/out
COPY --from=build --chown=coder:coder /tmp/conda-build /tmp/conda-build
COPY --from=build --chown=coder:coder /tmp/env_yaml /tmp/env_yaml
