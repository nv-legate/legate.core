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

# Usage: bind.sh --launcher <launcher> [--cpus <spec>] [--gpus <spec>] [--mems <spec>] [--nics <spec>] -- <app> ...
# <spec> specifies the resources to bind each node-local rank to, with ranks
# separated by /, e.g. 0,1/2,3/4,5/6,7 for 4 ranks per node.

set -euo pipefail

help() {
  echo "Usage: bind.sh -l | --launcher <mpirun,srun,jrun,local> [ -c | --cpus ] [ -g | --gpus ] [ -m | --mems ] [ -n | --nics ] -- <app>"
  exit 2
}

while :
do
  case "$1" in
    --launcher) launcher="$2" ;;
    --cpus) cpus="$2" ;;
    --gpus) gpus="$2" ;;
    --mems) mems="$2" ;;
    --nics) nics="$2" ;;
    --help) help ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      help
      ;;
  esac
  shift 2
done

if [ -z "$launcher" ]; then
  echo "bind.sh: -l / --launcher is a required argument"
fi

case "$launcher" in
  local) rank=0 ;;
  mpirun) rank="$OMPI_COMM_WORLD_LOCAL_RANK" ;;
  jsrun) rank="$OMPI_COMM_WORLD_LOCAL_RANK" ;;
  srun) rank="$SLURM_LOCALID" ;;
  *)
    echo "Unexpected launcher value: $launcher"
    help
    ;;
esac

export LEGATE_RANK="$rank"

if [ -n "${cpus+x}" ]; then
  cpus=(${cpus//\// })
  if [[ "$rank" -ge "${#cpus[@]}" ]]; then
      echo "Error: Incomplete CPU binding specification" 1>&2
      exit 1
  fi
fi

if [ -n "${gpus+x}" ]; then
  gpus=(${gpus//\// })
  if [[ "$rank" -ge "${#gpus[@]}" ]]; then
      echo "Error: Incomplete GPU binding specification" 1>&2
      exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${gpus[$rank]}"
fi

if [ -n "${mems+x}" ]; then
  mems=(${mems//\// })
  if [[ "$rank" -ge "${#mems[@]}" ]]; then
      echo "Error: Incomplete MEM binding specification" 1>&2
      exit 1
  fi
fi

if [ -n "${nics+x}" ]; then
  nics=(${nics//\// })
  if [[ "$rank" -ge "${#nics[@]}" ]]; then
      echo "Error: Incomplete NIC binding specification" 1>&2
      exit 1
  fi

  # Set all potentially relevant variables (hopefully they are ignored if we
  # are not using the corresponding network).
  nic="${nics[$rank]}"
  nic_array=(${nic//,/ })
  export UCX_NET_DEVICES="${nic//,/:1,}":1
  export NCCL_IB_HCA="$nic"
  export GASNET_NUM_QPS="${#nic_array[@]}"
  export GASNET_IBV_PORTS="${nic//,/+}"
fi

if [ $launcher != "local" ]; then
  if command -v numactl &> /dev/null; then
      if [[ -n "${cpus+x}" ]]; then
          set -- --physcpubind "${cpus[$rank]}" "$@"
      fi
      if [[ -n "${mems+x}" ]]; then
          set -- --membind "${mems[$rank]}" "$@"
      fi
      set -- numactl "$@"
  elif [[ -n "${cpus+x}" || -n "${mems+x}" ]]; then
      echo "Warning: numactl is not available, cannot bind to cores or memories" 1>&2
  fi
fi

exec "$@"
