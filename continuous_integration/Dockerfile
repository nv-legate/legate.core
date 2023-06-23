ARG BASE_IMAGE=rapidsai/devcontainers:23.06-cpp-cuda11.8-mambaforge-ubuntu22.04
FROM ${BASE_IMAGE} as setup

RUN export DEBIAN_FRONTEND=noninteractive       \
 && apt update                             \
 && apt install -y --no-install-recommends \
    rustc cargo

ENV PYTHONDONTWRITEBYTECODE=1
ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-east"
ENV SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev
ENV VAULT_HOST=https://vault.ops.k8s.rapids.ai
ENV HISTFILE=/home/coder/.cache/._bash_history

ARG USE_CUDA=ON
ENV USE_CUDA=${USE_CUDA}

ENV BUILD_MARCH=nocona

ENV PATH="${PATH}:/home/coder/.local/bin"

# ENV BOUNDS_CHECKS=OFF
# ENV GASNet_CONFIGURE_ARGS=--with-ibv-max-hcas=8
# ENV LEGATE_MAX_DIM=4
# ENV LEGATE_MAX_FIELDS=256
# ENV USE_HDF5=OFF
# ENV USE_LLVM=OFF
# ENV USE_OPENMP=ON
# ENV USE_SPY=OFF

USER coder

WORKDIR /home/coder/.cache
WORKDIR /home/coder

# Overwrite RAPIDS-AI devcontainer-utils-vault-s3-test
COPY continuous_integration/devcontainer-utils-vault-s3-test /usr/bin/devcontainer-utils-vault-s3-test

COPY --chown=coder:coder continuous_integration/home/coder/.gitconfig /home/coder/
COPY --chown=coder:coder continuous_integration/home/coder/.local/bin/* /home/coder/.local/bin/
COPY --chown=coder:coder . /home/coder/legate

RUN chmod a+x /home/coder/.local/bin/* && \
    mkdir -p /tmp/out && \
    chown -R coder:coder /tmp/out
    
RUN make-conda-env

#---------------------------------------------------
FROM setup as build

ARG AWS_SESSION_TOKEN
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

# If .creds exists copy it to /run/secrets
COPY --chown=coder:coder .cred[s] /run/secrets

RUN entrypoint build-all

#---------------------------------------------------
# Create an image without the secrets from above build stage
FROM setup as final
COPY --from=build /tmp/out /tmp/out
COPY --from=build /tmp/conda-build /tmp/conda-build
COPY --from=build /tmp/env_yaml /tmp/env_yaml


#---------------------------------------------------
FROM final as export-files
COPY --from=final /tmp/out out/
COPY --from=final /tmp/conda-build conda-build/



# ENTRYPOINT ["/home/coder/.local/bin/entrypoint"]

# CMD ["/bin/bash", "-l"]

