# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM rapidsai/rapidsai-dev:0.15-cuda11.0-devel-ubuntu18.04-py3.8

LABEL maintainer="Marcin Zalewski mzalewski@nvidia.com"

ARG INSTALL_ARGS=

WORKDIR /legate

COPY . /legate

WORKDIR /legate/legate.core

RUN /.run_in_rapids ./install.py --with-cuda /usr/local/cuda --cuda --openmp  --install-dir /legate/legate.core-install ${INSTALL_ARGS}
