ARG BASE_IMAGE=rapidsai/devcontainers:23.06-cpp-cuda11.8-mambaforge-ubuntu22.04
FROM ${BASE_IMAGE} as setup


ENV PYTHONDONTWRITEBYTECODE=1
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
FROM build as export-files
COPY --from=build /tmp/out out/
COPY --from=build /tmp/conda-build conda-build/



