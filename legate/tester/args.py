# Copyright 2022 NVIDIA Corporation
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
"""Provide an argparse ArgumentParser for the test runner.

"""
from __future__ import annotations

from argparse import ArgumentParser
from typing import Literal, Union

from typing_extensions import TypeAlias

from ..util.args import ExtendAction, MultipleChoices
from . import (
    DEFAULT_CPUS_PER_NODE,
    DEFAULT_GPU_DELAY,
    DEFAULT_GPU_MEMORY_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_NUMAMEM,
    DEFAULT_OMPS_PER_NODE,
    DEFAULT_OMPTHREADS,
    DEFAULT_RANKS_PER_NODE,
    FEATURES,
)

PinOptionsType: TypeAlias = Union[
    Literal["partial"],
    Literal["none"],
    Literal["strict"],
]

PIN_OPTIONS: tuple[PinOptionsType, ...] = (
    "partial",
    "none",
    "strict",
)


#: The argument parser for test.py
parser = ArgumentParser(
    description="Run the Cunumeric test suite",
    epilog="Any extra arguments will be forwarded to the Legate script",
)


stages = parser.add_argument_group("Feature stage selection")


stages.add_argument(
    "--use",
    dest="features",
    action=ExtendAction,
    choices=MultipleChoices(sorted(FEATURES)),
    type=lambda s: s.split(","),  # type: ignore [arg-type,return-value]
    help="Test Legate with features (also via USE_*)",
)


selection = parser.add_argument_group("Test file selection")


selection.add_argument(
    "--files",
    nargs="+",
    default=None,
    help="Explicit list of test files to run",
)


selection.add_argument(
    "--unit",
    dest="unit",
    action="store_true",
    default=False,
    help="Include unit tests",
)


feature_opts = parser.add_argument_group("Feature stage configuration options")


feature_opts.add_argument(
    "--cpus",
    dest="cpus",
    type=int,
    default=DEFAULT_CPUS_PER_NODE,
    help="Number of CPUs per node to use",
)


feature_opts.add_argument(
    "--gpus",
    dest="gpus",
    type=int,
    default=DEFAULT_GPUS_PER_NODE,
    help="Number of GPUs per node to use",
)


feature_opts.add_argument(
    "--omps",
    dest="omps",
    type=int,
    default=DEFAULT_OMPS_PER_NODE,
    help="Number OpenMP processors per node to use",
)


feature_opts.add_argument(
    "--utility",
    dest="utility",
    type=int,
    default=1,
    help="Number of of utility CPUs to reserve for runtime services",
)


feature_opts.add_argument(
    "--cpu-pin",
    dest="cpu_pin",
    choices=PIN_OPTIONS,
    default="partial",
    help="CPU pinning behavior on platforms that support CPU pinning",
)

feature_opts.add_argument(
    "--gpu-delay",
    dest="gpu_delay",
    type=int,
    default=DEFAULT_GPU_DELAY,
    help="Delay to introduce between GPU tests (ms)",
)


feature_opts.add_argument(
    "--fbmem",
    dest="fbmem",
    type=int,
    default=DEFAULT_GPU_MEMORY_BUDGET,
    help="GPU framebuffer memory (MB)",
)


feature_opts.add_argument(
    "--ompthreads",
    dest="ompthreads",
    metavar="THREADS",
    type=int,
    default=DEFAULT_OMPTHREADS,
    help="Number of threads per OpenMP processor",
)


feature_opts.add_argument(
    "--numamem",
    dest="numamem",
    type=int,
    default=DEFAULT_NUMAMEM,
    help="NUMA memory for OpenMP processors (MB)",
)

feature_opts.add_argument(
    "--ranks-per-node",
    dest="ranks",
    type=int,
    default=DEFAULT_RANKS_PER_NODE,
    help="Number of ranks per node to use",
)


test_opts = parser.add_argument_group("Test run configuration options")


test_opts.add_argument(
    "--timeout",
    dest="timeout",
    type=int,
    action="store",
    default=None,
    required=False,
    help="Timeout in seconds for individual tests",
)


test_opts.add_argument(
    "--legate",
    dest="legate_dir",
    metavar="LEGATE_DIR",
    action="store",
    default=None,
    required=False,
    help="Path to Legate installation directory",
)


test_opts.add_argument(
    "-C",
    "--directory",
    dest="test_root",
    metavar="DIR",
    action="store",
    default=None,
    required=False,
    help="Root directory containing the tests subdirectory",
)


test_opts.add_argument(
    "--cov-bin",
    default=None,
    help=(
        "coverage binary location, "
        "e.g. /conda_path/envs/env_name/bin/coverage"
    ),
)


test_opts.add_argument(
    "--cov-args",
    default="run -a --branch",
    help="coverage run command arguments, e.g. run -a --branch",
)


test_opts.add_argument(
    "--cov-src-path",
    default=None,
    help=(
        "path value of --source in coverage run command, "
        "e.g. /project_path/cunumeric/cunumeric"
    ),
)


test_opts.add_argument(
    "-j",
    "--workers",
    dest="workers",
    type=int,
    default=None,
    help="Number of parallel workers for testing",
)


test_opts.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="count",
    default=0,
    help="Display verbose output. Use -vv for even more output (test stdout)",
)


test_opts.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    help="Print the test plan but don't run anything",
)


test_opts.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Print out the commands that are to be executed",
)

parser.add_argument(
    "--color",
    dest="color",
    action="store_true",
    required=False,
    help="Whether to use color terminal output (if colorama is installed)",
)
