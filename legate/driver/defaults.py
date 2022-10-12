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

from os import environ, getcwd

__all__ = (
    "LEGATE_CPUS",
    "LEGATE_EAGER_ALLOC_PERCENTAGE",
    "LEGATE_FBMEM",
    "LEGATE_GPUS",
    "LEGATE_LOG_DIR",
    "LEGATE_NODES",
    "LEGATE_NUMAMEM",
    "LEGATE_OMP_PROCS",
    "LEGATE_OMP_THREADS",
    "LEGATE_RANKS_PER_NODE",
    "LEGATE_REGMEM",
    "LEGATE_SYSMEM",
    "LEGATE_UTILITY_CORES",
    "LEGATE_ZCMEM",
)

LEGATE_CPUS = 4
LEGATE_GPUS = 0
LEGATE_NODES = 1
LEGATE_RANKS_PER_NODE = 1

LEGATE_EAGER_ALLOC_PERCENTAGE = int(
    environ.get("LEGATE_EAGER_ALLOC_PERCENTAGE", 50)
)
LEGATE_FBMEM = int(environ.get("LEGATE_FBMEM", 4000))
LEGATE_LOG_DIR = getcwd()
LEGATE_NUMAMEM = int(environ.get("LEGATE_NUMAMEM", 0))
LEGATE_OMP_PROCS = int(environ.get("LEGATE_OMP_PROCS", 0))
LEGATE_OMP_THREADS = int(environ.get("LEGATE_OMP_THREADS", 4))
LEGATE_REGMEM = int(environ.get("LEGATE_REGMEM", 0))
LEGATE_SYSMEM = int(environ.get("LEGATE_SYSMEM", 4000))
LEGATE_UTILITY_CORES = int(environ.get("LEGATE_UTILITY_CORES", 2))
LEGATE_ZCMEM = int(environ.get("LEGATE_ZCMEM", 32))
