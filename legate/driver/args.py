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

from argparse import REMAINDER, ArgumentDefaultsHelpFormatter, ArgumentParser

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

multi_node = parser.add_argument_group("Multi-node configuration")
multi_node.add_argument(NODES.name, **NODES.kwargs)
multi_node.add_argument(RANKS_PER_NODE.name, **RANKS_PER_NODE.kwargs)
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
    help="profile Python execution with the cprofile module, "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nvprof",
    dest="nvprof",
    action="store_true",
    required=False,
    help="run Legate with nvprof, "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nsys",
    dest="nsys",
    action="store_true",
    required=False,
    help="run Legate with Nsight Systems, "
    "[legate-only, not supported with standard Python invocation]",
)


profiling.add_argument(
    "--nsys-targets",
    dest="nsys_targets",
    default="cublas,cuda,cudnn,nvtx,ucx",
    required=False,
    help="Specify profiling targets for Nsight Systems, "
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
    "(arguments with spaces inside must be additionally quoted), "
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
    help="run Legate inside gdb, "
    "[legate-only, not supported with standard Python invocation]",
)


debugging.add_argument(
    "--cuda-gdb",
    dest="cuda_gdb",
    action="store_true",
    required=False,
    help="run Legate inside cuda-gdb, "
    "[legate-only, not supported with standard Python invocation]",
)


debugging.add_argument(
    "--memcheck",
    dest="memcheck",
    action="store_true",
    required=False,
    help="run Legate with cuda-memcheck, "
    "[legate-only, not supported with standard Python invocation]",
)
debugging.add_argument(
    "--valgrind",
    dest="valgrind",
    action="store_true",
    required=False,
    help="run Legate with valgrind, "
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

debugging.add_argument(
    "--spy-validate-collective",
    dest="collective",
    action="store_true",
    required=False,
    help="Check that Legate made optimal use of collective views "
    "(developer option)",
)
debugging.add_argument(
    "--spy-assert-warning",
    dest="spy_assert_warning",
    action="store_true",
    required=False,
    help="Treat warnings as errors during processing of Legion Spy logs "
    "(developer option)",
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
    help="print out the final invocation run by bind.sh, "
    "[legate-only, not supported with standard Python invocation]",
)


other = parser.add_argument_group("Other options")


other.add_argument(
    "--module",
    dest="module",
    default=None,
    required=False,
    help="Specify a Python module to load before running, "
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
    help="Whether to run with rlwrap to improve readline ability, "
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
    help="Print information about the capabilities of this build of legate, "
    "and immediately exit.",
)
