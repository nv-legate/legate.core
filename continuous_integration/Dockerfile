FROM rapidsai/devcontainers:23.06-cpp-cuda11.8-mambaforge-ubuntu22.04

RUN export DEBIAN_FRONTEND=noninteractive       \
 && sudo apt update                             \
 && sudo apt install -y --no-install-recommends \
    tree vim wget rustc cargo

ENV PYTHONDONTWRITEBYTECODE=1
ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev
ENV VAULT_HOST=https://vault.ops.k8s.rapids.ai
ENV HISTFILE=/home/coder/.cache/._bash_history

ENV USE_CUDA=ON

ENV BUILD_MARCH=nocona

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

ENTRYPOINT ["/home/coder/.local/bin/entrypoint"]

CMD ["/bin/bash", "-l"]

