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

help() {
  cat 1>&2 <<EOM
Usage: bind.sh [OPTIONS]... -- APP...

Options:
  --launcher {mpirun|srun|jrun|auto|local}
                    Launcher type, used to set LEGATE_RANK
                    If 'auto', attempt to find the launcher rank automatically
                    If 'local', rank is set to "0".
  --cpus SPEC       CPU binding specification, passed to numactl
  --gpus SPEC       GPU binding specification, used to set CUDA_VISIBLE_DEVICES
  --mems SPEC       Memory binding specification, passed to numactl
  --nics SPEC       Network interface binding specification, used to set
                    all of: UCX_NET_DEVICES, NCCL_IB_HCA, GASNET_NUM_QPS,
                    and GASNET_IBV_PORTS
  --debug           print out the final computed invocation before exectuting

SPEC specifies the resources to bind each node-local rank to, with ranks
separated by /, e.g. '0,1/2,3/4,5/6,7' for 4 ranks per node.

APP is the application that will be executed by bind.sh, as well as any
arguments for it.

If --cpus or --mems is specified, then APP will be invoked with numactl.

An explicit '--' separator should always come after OPTIONS and before APP.
EOM
  exit 2
}

debug="0"
launcher=auto
while :
do
  case "$1" in
    --launcher) launcher="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --mems) mems="$2"; shift 2 ;;
    --nics) nics="$2"; shift 2 ;;
    --debug) debug="1"; shift ;;
    --help) help ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1" 1>&2
      help
      ;;
  esac
done

case "$launcher" in
  mpirun)
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-unknown}"
    global_rank="${OMPI_COMM_WORLD_RANK:-unknown}"
    ;;
  jsrun )
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-unknown}"
    global_rank="${OMPI_COMM_WORLD_RANK:-unknown}"
    ;;
  srun  )
    local_rank="${SLURM_LOCALID:-unknown}"
    global_rank="${SLURM_PROCID:-unknown}"
    ;;
  auto  )
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-unknown}}}"
    global_rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${MV2_COMM_WORLD_RANK:-${SLURM_PROCID:-unknown}}}}"
    ;;
  local )
    local_rank="0"
    global_rank="0"
    ;;
  *)
    echo "Unexpected launcher value: $launcher" 1>&2
    help
    ;;
esac

if [[ "$local_rank" == "unknown" ]]; then
    echo "Error: Could not determine node-local rank" 1>&2
    exit 1
fi

if [[ "$global_rank" == "unknown" ]]; then
    echo "Error: Could not determine global rank" 1>&2
    exit 1
fi

export LEGATE_LOCAL_RANK="$local_rank"
export LEGATE_GLOBAL_RANK="$global_rank"

if [ -n "${cpus+x}" ]; then
  cpus=(${cpus//\// })
  if [[ "$local_rank" -ge "${#cpus[@]}" ]]; then
      echo "Error: Incomplete CPU binding specification" 1>&2
      exit 1
  fi
fi

if [ -n "${gpus+x}" ]; then
  gpus=(${gpus//\// })
  if [[ "$local_rank" -ge "${#gpus[@]}" ]]; then
      echo "Error: Incomplete GPU binding specification" 1>&2
      exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${gpus[$local_rank]}"
fi

if [ -n "${mems+x}" ]; then
  mems=(${mems//\// })
  if [[ "$local_rank" -ge "${#mems[@]}" ]]; then
      echo "Error: Incomplete MEM binding specification" 1>&2
      exit 1
  fi
fi

if [ -n "${nics+x}" ]; then
  nics=(${nics//\// })
  if [[ "$local_rank" -ge "${#nics[@]}" ]]; then
      echo "Error: Incomplete NIC binding specification" 1>&2
      exit 1
  fi

  # set all potentially relevant variables (hopefully they are ignored if we
  # are not using the corresponding network)
  nic="${nics[$local_rank]}"
  nic_array=(${nic//,/ })
  export UCX_NET_DEVICES="${nic//,/:1,}":1
  export GASNET_NUM_QPS="${#nic_array[@]}"
  export GASNET_IBV_PORTS="${nic//,/+}"

  # NCCL is supposed to detect the topology and use the right NIC automatically.
  # NCCL env vars must be set the same way for all ranks on the same node, so
  # the best we can do here is to constrain NCCL to the full set of NICs that
  # the user specified.
  # Note the added "=", to do exact instead of prefix match.
  export NCCL_IB_HCA="=$(IFS=, ; echo "${nics[*]}")"
fi

# numactl is only needed if cpu or memory pinning was requested
if [[ -n "${cpus+x}" || -n "${mems+x}" ]]; then
  if command -v numactl &> /dev/null; then
      if [[ -n "${cpus+x}" ]]; then
          set -- --physcpubind "${cpus[$local_rank]}" "$@"
      fi
      if [[ -n "${mems+x}" ]]; then
          set -- --membind "${mems[$local_rank]}" "$@"
      fi
      set -- numactl "$@"
  else
      echo "Warning: numactl is not available, cannot bind to cores or memories" 1>&2
  fi
fi

# arguments may contain the substring %%LEGATE_GLOBAL_RANK%% which needs to be
# be replaced with the actual computed rank for downstream processes to use
updated=()
for arg in "$@"; do
  updated+=("${arg/\%\%LEGATE_GLOBAL_RANK\%\%/$LEGATE_GLOBAL_RANK}")
done

set -- "${updated[@]}"

if [ "$debug" == "1" ]; then
  echo -n "bind.sh:"
  for TOK in "$@"; do printf " %q" "$TOK"; done
  echo
fi

exec "$@"
