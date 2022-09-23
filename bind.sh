#!/bin/bash

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

set -euo pipefail

# Usage: bind.sh <launcher> [--cpus <spec>] [--gpus <spec>] [--mems <spec>] [--nics <spec>] <app> ...
# <spec> specifies the resources to bind each node-local rank to, with ranks
# separated by /, e.g. 0,1/2,3/4,5/6,7 for 4 ranks per node.

# Detect node-local rank based on launcher
IDX=none
case "$1" in
    mpirun) IDX="$OMPI_COMM_WORLD_LOCAL_RANK" ;;
    jsrun) IDX="$OMPI_COMM_WORLD_LOCAL_RANK" ;;
    srun) IDX="$SLURM_LOCALID" ;;
    local) IDX=0 ;;
    none) IDX="${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-none}}}" ;;
esac
shift
if [[ "$IDX" == "none" ]]; then
    echo "Error: Cannot detect node-local rank" 1>&2
    exit 1
fi

# Read binding specifications
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpus)
            CPUS=(${2//\// })
            if [[ "$IDX" -ge "${#CPUS[@]}" ]]; then
                echo "Error: Incomplete CPU binding specification" 1>&2
                exit 1
            fi
            ;;
        --gpus)
            GPUS=(${2//\// })
            if [[ "$IDX" -ge "${#GPUS[@]}" ]]; then
                echo "Error: Incomplete GPU binding specification" 1>&2
                exit 1
            fi
            ;;
        --mems)
            MEMS=(${2//\// })
            if [[ "$IDX" -ge "${#MEMS[@]}" ]]; then
                echo "Error: Incomplete MEM binding specification" 1>&2
                exit 1
            fi
            ;;
        --nics)
            NICS=(${2//\// })
            if [[ "$IDX" -ge "${#NICS[@]}" ]]; then
                echo "Error: Incomplete NIC binding specification" 1>&2
                exit 1
            fi
            ;;
        *)
            break
            ;;
    esac
    shift 2
done

# Prepare environment
if [[ -n "${GPUS+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS[$IDX]}"
fi
if [[ -n "${NICS+x}" ]]; then
    # Set all potentially relevant variables, hopefully they are ignored if we
    # are not using the corresponding network.
    NIC="${NICS[$IDX]}"
    export UCX_NET_DEVICES="${NIC//,/:1,}":1
    export NCCL_IB_HCA="$NIC"
    NIC_ARR=(${NIC//,/ })
    export GASNET_NUM_QPS="${#NIC_ARR[@]}"
    export GASNET_IBV_PORTS="${NIC//,/+}"
fi

# Prepare command
if command -v numactl &> /dev/null; then
    if [[ -n "${CPUS+x}" ]]; then
        set -- --physcpubind "${CPUS[$IDX]}" "$@"
    fi
    if [[ -n "${MEMS+x}" ]]; then
        set -- --membind "${MEMS[$IDX]}" "$@"
    fi
    set -- numactl "$@"
elif [[ -n "${CPUS+x}" || -n "${MEMS+x}" ]]; then
    echo "Warning: numactl is not available, cannot bind to cores or memories" 1>&2
fi
exec "$@"
