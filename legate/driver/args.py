#!/usr/bin/env python

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
from __future__ import annotations

import re
from argparse import REMAINDER, ArgumentDefaultsHelpFormatter, ArgumentParser
from os import getenv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from .. import __version__
from ..util.args import InfoAction
from ..util.shared_args import (
    CPUS,
    FBMEM,
    GPUS,
    LAUNCHER,
    LAUNCHER_EXTRA,
    NOCR,
    NODES,
    NUMAMEM,
    OMPS,
    OMPTHREADS,
    RANKS_PER_NODE,
    REGMEM,
    SYSMEM,
    UTILITY,
    ZCMEM,
)
from . import defaults

__all__ = ("parser",)


def _get_ompi_config() -> tuple[int, int] | None:
    if not (ranks_env := getenv("OMPI_COMM_WORLD_SIZE")):
        return None

    if not (ranks_per_node_env := getenv("OMPI_COMM_WORLD_LOCAL_SIZE")):
        return None

    try:
        ranks, ranks_per_node = int(ranks_env), int(ranks_per_node_env)
    except ValueError:
        raise ValueError(
            "Expected OMPI_COMM_WORLD_SIZE and OMPI_COMM_WORLD_LOCAL_SIZE to "
            f"be integers, got OMPI_COMM_WORLD_SIZE={ranks_env} and "
            f"OMPI_COMM_WORLD_LOCAL_SIZE={ranks_per_node_env}"
        )

    if ranks % ranks_per_node != 0:
        raise ValueError(
            "Detected incompatible ranks and ranks-per-node from "
            f"OMPI_COMM_WORLD_SIZE={ranks} and "
            f"OMPI_COMM_WORLD_LOCAL_SIZE={ranks_per_node}"
        )

    return ranks // ranks_per_node, ranks_per_node


def _get_mv2_config() -> tuple[int, int] | None:
    if not (ranks_env := getenv("MV2_COMM_WORLD_SIZE")):
        return None

    if not (ranks_per_node_env := getenv("MV2_COMM_WORLD_LOCAL_SIZE")):
        return None

    try:
        ranks, ranks_per_node = int(ranks_env), int(ranks_per_node_env)
    except ValueError:
        raise ValueError(
            "Expected MV2_COMM_WORLD_SIZE and MV2_COMM_WORLD_LOCAL_SIZE to "
            f"be integers, got MV2_COMM_WORLD_SIZE={ranks_env} and "
            f"MV2_COMM_WORLD_LOCAL_SIZE={ranks_per_node_env}"
        )

    if ranks % ranks_per_node != 0:
        raise ValueError(
            "Detected incompatible ranks and ranks-per-node from "
            f"MV2_COMM_WORLD_SIZE={ranks} and "
            f"MV2_COMM_WORLD_LOCAL_SIZE={ranks_per_node}"
        )

    return ranks // ranks_per_node, ranks_per_node


_SLURM_CONFIG_ERROR = (
    "Expected SLURM_TASKS_PER_NODE to be a single integer ranks per node, or "
    "of the form 'A(xB)' where A is an integer ranks per node, and B is an "
    "integer number of nodes, got SLURM_TASKS_PER_NODE={value}"
)


def _get_slurm_config() -> tuple[int, int] | None:
    if not (nodes_env := getenv("SLURM_JOB_NUM_NODES")):
        return None

    nprocs_env = getenv("SLURM_NPROCS")
    ntasks_env = getenv("SLURM_NTASKS")
    tasks_per_node_env = getenv("SLURM_TASKS_PER_NODE")

    # at least one of these needs to be set
    if not any((nprocs_env, ntasks_env, tasks_per_node_env)):
        return None

    # use SLURM_TASKS_PER_NODE if it is given
    if tasks_per_node_env is not None:
        try:
            return 1, int(tasks_per_node_env)
        except ValueError:
            m = re.match(r"^(\d*)\(x(\d*)\)$", tasks_per_node_env.strip())
            if m:
                try:
                    return int(m.group(2)), int(m.group(1))
                except ValueError:
                    pass
            raise ValueError(
                _SLURM_CONFIG_ERROR.format(value=tasks_per_node_env)
            )

    # prefer newer SLURM_NTASKS over SLURM_NPROCS
    if ntasks_env is not None:
        try:
            nodes, ranks = int(nodes_env), int(ntasks_env)
        except ValueError:
            raise ValueError(
                "Expected SLURM_JOB_NUM_NODES and SLURM_NTASKS to "
                f"be integers, got SLURM_JOB_NUM_NODES={nodes_env} and "
                f"SLURM_NTASKS={ntasks_env}"
            )

        if ranks % nodes != 0:
            raise ValueError(
                "Detected incompatible ranks and ranks-per-node from "
                f"SLURM_JOB_NUM_NODES={nodes} and "
                f"SLURM_NTASKS={ranks}"
            )

        return nodes, ranks // nodes

    # fall back to older SLURM_NPROCS
    if nprocs_env is not None:
        try:
            nodes, ranks = int(nodes_env), int(nprocs_env)
        except ValueError:
            raise ValueError(
                "Expected SLURM_JOB_NUM_NODES and SLURM_NPROCS to "
                f"be integers, got SLURM_JOB_NUM_NODES={nodes_env} and "
                f"SLURM_NPROCS={nprocs_env}"
            )

        if ranks % nodes != 0:
            raise ValueError(
                "Detected incompatible ranks and ranks-per-node from "
                f"SLURM_JOB_NUM_NODES={nodes} and "
                f"SLURM_NPROCS={ranks}"
            )

        return nodes, ranks // nodes

    return None


def detect_multi_node_defaults() -> tuple[dict[str, Any], dict[str, Any]]:
    nodes_kw = dict(NODES.kwargs)
    ranks_per_node_kw = dict(RANKS_PER_NODE.kwargs)
    where = None

    if config := _get_ompi_config():
        where = "OMPI"
    elif config := _get_mv2_config():
        where = "MV2"
    elif config := _get_slurm_config():
        where = "SLURM"
    else:
        config = defaults.LEGATE_NODES, defaults.LEGATE_RANKS_PER_NODE
        where = None

    nodes, ranks_per_node = config
    nodes_kw["default"] = nodes
    ranks_per_node_kw["default"] = ranks_per_node

    if where:
        extra = f" [default auto-detected from {where}]"
        nodes_kw["help"] += extra
        ranks_per_node_kw["help"] += extra

    return nodes_kw, ranks_per_node_kw


parser = ArgumentParser(
    description="Legate Driver",
    allow_abbrev=False,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "command",
    nargs=REMAINDER,
    help="A python script to run, plus any arguments for the script. "
    "Any arguments after the script will be passed to the script, i.e. "
    "NOT used as arguments to legate itself.",
)

nodes_kw, ranks_per_node_kw = detect_multi_node_defaults()


multi_node = parser.add_argument_group("Multi-node configuration")
multi_node.add_argument(NODES.name, **nodes_kw)
multi_node.add_argument(RANKS_PER_NODE.name, **ranks_per_node_kw)
multi_node.add_argument(NOCR.name, **NOCR.kwargs)
multi_node.add_argument(LAUNCHER.name, **LAUNCHER.kwargs)
multi_node.add_argument(LAUNCHER_EXTRA.name, **LAUNCHER_EXTRA.kwargs)


binding = parser.add_argument_group("Hardware binding")


binding.add_argument(
    "--cpu-bind",
    help="CPU cores to bind each rank to. Comma-separated core IDs as "
    "well as ranges are accepted, as reported by `numactl`. Binding "
    "instructions for all ranks should be listed in one string, separated "
    "by `/`. "
    "[legate-only, not supported with standard Python invocation]",
)


binding.add_argument(
    "--mem-bind",
    help="NUMA memories to bind each rank to. Use comma-separated integer "
    "IDs as reported by `numactl`. Binding instructions for all ranks "
    "should be listed in one string, separated by `/`. "
    "[legate-only, not supported with standard Python invocation]",
)


binding.add_argument(
    "--gpu-bind",
    help="GPUs to bind each rank to. Use comma-separated integer IDs as "
    "reported by `nvidia-smi`. Binding instructions for all ranks "
    "should be listed in one string, separated by `/`. "
    "[legate-only, not supported with standard Python invocation]",
)


binding.add_argument(
    "--nic-bind",
    help="NICs to bind each rank to. Use comma-separated device names as "
    "appropriate for the network in use. Binding instructions for all ranks "
    "should be listed in one string, separated by `/`. "
    "[legate-only, not supported with standard Python invocation]",
)


core = parser.add_argument_group("Core allocation")
core.add_argument(CPUS.name, **CPUS.kwargs)
core.add_argument(GPUS.name, **GPUS.kwargs)
core.add_argument(OMPS.name, **OMPS.kwargs)
core.add_argument(OMPTHREADS.name, **OMPTHREADS.kwargs)
core.add_argument(UTILITY.name, **UTILITY.kwargs)


memory = parser.add_argument_group("Memory allocation")
memory.add_argument(SYSMEM.name, **SYSMEM.kwargs)
memory.add_argument(NUMAMEM.name, **NUMAMEM.kwargs)
memory.add_argument(FBMEM.name, **FBMEM.kwargs)
memory.add_argument(ZCMEM.name, **ZCMEM.kwargs)
memory.add_argument(REGMEM.name, **REGMEM.kwargs)


# FIXME: We set the eager pool size to 50% of the total size for now.
#        This flag will be gone once we roll out a new allocation scheme.
memory.add_argument(
    "--eager-alloc-percentage",
    dest="eager_alloc",
    default=defaults.LEGATE_EAGER_ALLOC_PERCENTAGE,
    required=False,
    help="Specify the size of eager allocation pool in percentage",
)


profiling = parser.add_argument_group("Profiling")


profiling.add_argument(
    "--profile",
    dest="profile",
    action="store_true",
    required=False,
    help="profile Legate execution",
)


profiling.add_argument(
    "--cprofile",
    dest="cprofile",
    action="store_true",
    required=False,
    help="profile Python execution with the cprofile module "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nvprof",
    dest="nvprof",
    action="store_true",
    required=False,
    help="run Legate with nvprof "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nsys",
    dest="nsys",
    action="store_true",
    required=False,
    help="run Legate with Nsight Systems "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nsys-targets",
    dest="nsys_targets",
    default="cublas,cuda,cudnn,nvtx,ucx",
    required=False,
    help="Specify profiling targets for Nsight Systems "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nsys-extra",
    dest="nsys_extra",
    action="append",
    default=[],
    required=False,
    help="Specify extra flags for Nsight Systems (can appear more than once). "
    "Multiple arguments may be provided together in a quoted string "
    "(arguments with spaces inside must be additionally quoted) "
    "[legate-only, not supported with standard Python invocation]",
)

logging = parser.add_argument_group("Logging")


logging.add_argument(
    "--logging",
    type=str,
    default=None,
    dest="user_logging_levels",
    help="extra loggers to turn on",
)

logging.add_argument(
    "--logdir",
    type=str,
    default=defaults.LEGATE_LOG_DIR,
    dest="logdir",
    help="Directory for Legate log files (automatically created if it "
    "doesn't exist; defaults to current directory if not set)",
)


logging.add_argument(
    "--log-to-file",
    dest="log_to_file",
    action="store_true",
    required=False,
    help="redirect logging output to a file inside --logdir",
)


debugging = parser.add_argument_group("Debugging")


debugging.add_argument(
    "--gdb",
    dest="gdb",
    action="store_true",
    required=False,
    help="run Legate inside gdb "
    "[legate-only, not supported with standard Python invocation]",
)


debugging.add_argument(
    "--cuda-gdb",
    dest="cuda_gdb",
    action="store_true",
    required=False,
    help="run Legate inside cuda-gdb "
    "[legate-only, not supported with standard Python invocation]",
)


debugging.add_argument(
    "--memcheck",
    dest="memcheck",
    action="store_true",
    required=False,
    help="run Legate with cuda-memcheck "
    "[legate-only, not supported with standard Python invocation]",
)
debugging.add_argument(
    "--valgrind",
    dest="valgrind",
    action="store_true",
    required=False,
    help="run Legate with valgrind "
    "[legate-only, not supported with standard Python invocation]",
)


debugging.add_argument(
    "--freeze-on-error",
    dest="freeze_on_error",
    action="store_true",
    required=False,
    help="if the program crashes, freeze execution right before exit so a "
    "debugger can be attached",
)


debugging.add_argument(
    "--gasnet-trace",
    dest="gasnet_trace",
    action="store_true",
    default=False,
    required=False,
    help="enable GASNet tracing (assumes GASNet was configured with "
    "--enable-trace)",
)


debugging.add_argument(
    "--spy",
    dest="spy",
    action="store_true",
    required=False,
    help="Generate Legion Spy logs",
)


info = parser.add_argument_group("Informational")


info.add_argument(
    "--progress",
    dest="progress",
    action="store_true",
    required=False,
    help="show progress of operations when running the program",
)


info.add_argument(
    "--mem-usage",
    dest="mem_usage",
    action="store_true",
    required=False,
    help="report the memory usage by Legate in every memory",
)


info.add_argument(
    "--verbose",
    dest="verbose",
    action="store_true",
    required=False,
    help="print out each shell command before running it",
)


info.add_argument(
    "--bind-detail",
    dest="bind_detail",
    action="store_true",
    required=False,
    help="print out the final invocation run by bind.sh "
    "[legate-only, not supported with standard Python invocation]",
)


other = parser.add_argument_group("Other options")

other.add_argument(
    "--timing",
    dest="timing",
    action="store_true",
    required=False,
    help="Print overall process start and end timestamps to stdout "
    "[legate-only, not supported with standard Python invocation]",
)

other.add_argument(
    "--wrapper",
    dest="wrapper",
    required=False,
    action="append",
    default=[],
    help="Specify another executable (and any command-line arguments for that "
    "executable) to wrap the Legate executable invocation. This wrapper will "
    "come right after the launcher invocation, and will be passed the rest of "
    "the Legate invocation (including any other wrappers) to execute. May "
    "contain the special string %%%%LEGATE_GLOBAL_RANK%%%% that will be "
    "replaced with the rank of the current process by bind.sh. If multiple "
    "--wrapper values are provided, they will execute in the order given. "
    "[legate-only, not supported with standard Python invocation]",
)

other.add_argument(
    "--wrapper-inner",
    dest="wrapper_inner",
    required=False,
    action="append",
    default=[],
    help="Specify another executable (and any command-line arguments for that "
    "executable) to wrap the Legate executable invocation. This wrapper will "
    "come right before the legion_python invocation (after any other "
    "wrappers) and will be passed the rest of the legion_python invocation to "
    "execute. May contain the special string %%%%LEGATE_GLOBAL_RANK%%%% that "
    "will be replaced with the rank of the current process by bind.sh. If "
    "multiple --wrapper-inner values are given, they will execute in the "
    "order given. "
    "[legate-only, not supported with standard Python invocation]",
)

other.add_argument(
    "--module",
    dest="module",
    default=None,
    required=False,
    help="Specify a Python module to load before running "
    "[legate-only, not supported with standard Python invocation]",
)


other.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    required=False,
    help="Print the full command line invocation that would be "
    "executed, without executing it",
)

other.add_argument(
    "--rlwrap",
    dest="rlwrap",
    action="store_true",
    required=False,
    help="Whether to run with rlwrap to improve readline ability "
    "[legate-only, not supported with standard Python invocation]",
)

other.add_argument(
    "--color",
    dest="color",
    action="store_true",
    required=False,
    help="Whether to use color terminal output (if colorama is installed)",
)

other.add_argument(
    "--version",
    action="version",
    version=__version__,
)

other.add_argument(
    "--info",
    action=InfoAction,
    help="Print information about the capabilities of this build of legate "
    "and immediately exit.",
)
