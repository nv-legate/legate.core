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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from . import defaults
from .types import LauncherType

__all__ = ("parser",)

LAUNCHERS: tuple[LauncherType, ...] = ("mpirun", "jsrun", "srun", "none")

parser = ArgumentParser(
    description="Legate Driver",
    allow_abbrev=False,
    formatter_class=ArgumentDefaultsHelpFormatter,
)


multi_node = parser.add_argument_group("Multi-node configuration")


multi_node.add_argument(
    "--nodes",
    type=int,
    default=defaults.LEGATE_NODES,
    dest="nodes",
    help="Number of nodes to use",
)


multi_node.add_argument(
    "--ranks-per-node",
    type=int,
    default=defaults.LEGATE_RANKS_PER_NODE,
    dest="ranks_per_node",
    help="Number of ranks (processes running copies of the program) to "
    "launch per node. The default (1 rank per node) will typically result "
    "in the best performance.",
)


multi_node.add_argument(
    "--no-replicate",
    dest="not_control_replicable",
    action="store_true",
    required=False,
    help="Execute this program without control replication.  Most of the "
    "time, this is not recommended.  This option should be used for "
    "debugging.  The -lg:safe_ctrlrepl Legion option may be helpful "
    "with discovering issues with replicated control.",
)

multi_node.add_argument(
    "--launcher",
    dest="launcher",
    choices=LAUNCHERS,
    default="none",
    help='launcher program to use (set to "none" for local runs, or if '
    "the launch has already happened by the time legate is invoked)",
)


multi_node.add_argument(
    "--launcher-extra",
    dest="launcher_extra",
    action="append",
    default=[],
    required=False,
    help="additional argument to pass to the launcher (can appear more "
    "than once)",
)


binding = parser.add_argument_group("Hardware binding")


binding.add_argument(
    "--cpu-bind",
    help="CPU cores to bind each rank to. Comma-separated core IDs as "
    "well as ranges are accepted, as reported by `numactl`. Binding "
    "instructions for all ranks should be listed in one string, separated "
    "by `/`.",
)


binding.add_argument(
    "--mem-bind",
    help="NUMA memories to bind each rank to. Use comma-separated integer "
    "IDs as reported by `numactl`. Binding instructions for all ranks "
    "should be listed in one string, separated by `/`.",
)


binding.add_argument(
    "--gpu-bind",
    help="GPUs to bind each rank to. Use comma-separated integer IDs as "
    "reported by `nvidia-smi`. Binding instructions for all ranks "
    "should be listed in one string, separated by `/`.",
)


binding.add_argument(
    "--nic-bind",
    help="NICs to bind each rank to. Use comma-separated device names as "
    "appropriate for the network in use. Binding instructions for all ranks "
    "should be listed in one string, separated by `/`.",
)


core = parser.add_argument_group("Core alloction")


core.add_argument(
    "--cpus",
    type=int,
    default=defaults.LEGATE_CPUS,
    dest="cpus",
    help="Number of CPUs to use per rank",
)


core.add_argument(
    "--gpus",
    type=int,
    default=defaults.LEGATE_GPUS,
    dest="gpus",
    help="Number of GPUs to use per rank",
)


core.add_argument(
    "--omps",
    type=int,
    default=defaults.LEGATE_OMP_PROCS,
    dest="openmp",
    help="Number of OpenMP groups to use per rank",
)


core.add_argument(
    "--ompthreads",
    type=int,
    default=defaults.LEGATE_OMP_THREADS,
    dest="ompthreads",
    help="Number of threads per OpenMP group",
)


core.add_argument(
    "--utility",
    type=int,
    default=defaults.LEGATE_UTILITY_CORES,
    dest="utility",
    help="Number of Utility processors per rank to request for meta-work",
)


memory = parser.add_argument_group("Memory alloction")

memory.add_argument(
    "--sysmem",
    type=int,
    default=defaults.LEGATE_SYSMEM,
    dest="sysmem",
    help="Amount of DRAM memory per rank (in MBs)",
)


memory.add_argument(
    "--numamem",
    type=int,
    default=defaults.LEGATE_NUMAMEM,
    dest="numamem",
    help="Amount of DRAM memory per NUMA domain per rank (in MBs)",
)


memory.add_argument(
    "--fbmem",
    type=int,
    default=defaults.LEGATE_FBMEM,
    dest="fbmem",
    help="Amount of framebuffer memory per GPU (in MBs)",
)


memory.add_argument(
    "--zcmem",
    type=int,
    default=defaults.LEGATE_ZCMEM,
    dest="zcmem",
    help="Amount of zero-copy memory per rank (in MBs)",
)


memory.add_argument(
    "--regmem",
    type=int,
    default=defaults.LEGATE_REGMEM,
    dest="regmem",
    help="Amount of registered CPU-side pinned memory per rank (in MBs)",
)


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
    "--nvprof",
    dest="nvprof",
    action="store_true",
    required=False,
    help="run Legate with nvprof",
)


profiling.add_argument(
    "--nsys",
    dest="nsys",
    action="store_true",
    required=False,
    help="run Legate with Nsight Systems",
)


profiling.add_argument(
    "--nsys-targets",
    dest="nsys_targets",
    default="cublas,cuda,cudnn,nvtx,ucx",
    required=False,
    help="Specify profiling targets for Nsight Systems",
)


profiling.add_argument(
    "--nsys-extra",
    dest="nsys_extra",
    action="append",
    default=[],
    required=False,
    help="Specify extra flags for Nsight Systems",
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


logging.add_argument(
    "--keep-logs",
    dest="keep_logs",
    action="store_true",
    required=False,
    help="don't delete profiler & spy dumps after processing",
)


debugging = parser.add_argument_group("Debugging")


debugging.add_argument(
    "--gdb",
    dest="gdb",
    action="store_true",
    required=False,
    help="run Legate inside gdb",
)


debugging.add_argument(
    "--cuda-gdb",
    dest="cuda_gdb",
    action="store_true",
    required=False,
    help="run Legate inside cuda-gdb",
)


debugging.add_argument(
    "--memcheck",
    dest="memcheck",
    action="store_true",
    required=False,
    help="run Legate with cuda-memcheck",
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
    "--enable--trace)",
)

debugging.add_argument(
    "--dataflow",
    dest="dataflow",
    action="store_true",
    required=False,
    help="Generate Legate dataflow graph",
)


debugging.add_argument(
    "--event",
    dest="event",
    action="store_true",
    required=False,
    help="Generate Legate event graph",
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


other = parser.add_argument_group("Other options")


other.add_argument(
    "--module",
    dest="module",
    default=None,
    required=False,
    help="Specify a Python module to load before running",
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
    help="Whether to run with rlwrap to improve readline ability",
)
