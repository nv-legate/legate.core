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
"""Utilities and helpers for implementing the Cunumeric custom test runner.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from typing_extensions import Literal, TypeAlias

from ..util.types import ArgList

#: Define the available feature types for tests
FeatureType: TypeAlias = Union[
    Literal["cpus"], Literal["cuda"], Literal["eager"], Literal["openmp"]
]

#: Value to use if --cpus is not specified.
DEFAULT_CPUS_PER_NODE = 2

#: Value to use if --gpus is not specified.
DEFAULT_GPUS_PER_NODE = 1

# Value to use if --bloat-factor is not specified
DEFAULT_GPU_BLOAT_FACTOR = 1.5

# Delay to introduce between GPU test invocations (ms)
DEFAULT_GPU_DELAY = 2000

# Value to use if --fbmem is not specified (MB)
DEFAULT_GPU_MEMORY_BUDGET = 4096

#: Value to use if --omps is not specified.
DEFAULT_OMPS_PER_NODE = 1

#: Value to use if --ompthreads is not specified.
DEFAULT_OMPTHREADS = 4

#: Value to use if --numamem is not specified.
DEFAULT_NUMAMEM = 0

#: Value to use if --ranks-per-node is not specified.
DEFAULT_RANKS_PER_NODE = 1

#: Default values to apply to normalize the testing environment.
DEFAULT_PROCESS_ENV = {
    "LEGATE_TEST": "1",
}

#: Feature values that are accepted for --use, in the relative order
#: that the corresponding test stages should always execute in
FEATURES: tuple[FeatureType, ...] = (
    "cpus",
    "cuda",
    "eager",
    "openmp",
)

#: Paths to test files that should be skipped entirely in all stages.
#:
#: Client test scripts should udpate this set with their own customizations.
SKIPPED_EXAMPLES: set[str] = set()

#: Extra arguments to add when specific test files are executed (in any stage).
#:
#: Client test scripts should udpate this dict with their own customizations.
PER_FILE_ARGS: dict[str, ArgList] = {}


@dataclass
class CustomTest:
    file: str
    kind: FeatureType
    args: ArgList


#: Customized configurations for specific test files. Each entry will result
#: in the specified test file being run in the specified stage, with the given
#: command line arguments appended (overriding default stage arguments). These
#: files are run serially, after the sharded, parallelized tests.
#:
#: Client test scripts should udpate this set with their own customizations.
CUSTOM_FILES: list[CustomTest] = []

#: Location to store a list of last-failed tests
LAST_FAILED_FILENAME: str = ".legate-test-last-failed"
