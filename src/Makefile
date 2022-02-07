# Copyright 2021-2022 NVIDIA Corporation
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

ifndef LEGATE_DIR
$(error LEGATE_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG    ?= 0  # Include debugging symbols
GPU_ARCH ?= volta
LIBNAME   = liblgcore
USE_GPU_REDUCTIONS = 0

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
INC_FLAGS	+= -I$(shell pwd)

CC_FLAGS ?=

NVCC_FLAGS ?=

LD_FLAGS ?=

# We also need cuda init for runs with GPUs
ifeq ($(strip $(USE_CUDA)),1)
ifndef CUDA
$(error CUDA variable is not defined, aborting build)
endif
endif

BOOTSTRAP = 1

include core.mk

# We can just include directly since we know we're the core Makefile
include legate.mk
