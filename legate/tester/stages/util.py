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
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, Union

from typing_extensions import TypeAlias

from ...util.ui import failed, passed, shell, skipped, timeout
from ..config import Config
from ..logger import LOG
from ..test_system import ProcessResult

UNPIN_ENV = {"REALM_SYNTHETIC_CORE_MAP": ""}

# Raise min chunk sizes for deferred codepaths to force eager execution
EAGER_ENV = {
    "CUNUMERIC_FORCE_THUNK": "eager",
    "CUNUMERIC_MIN_CPU_CHUNK": "2000000000",
    "CUNUMERIC_MIN_OMP_CHUNK": "2000000000",
    "CUNUMERIC_MIN_GPU_CHUNK": "2000000000",
}


RankShard: TypeAlias = Tuple[int, ...]


@dataclass(frozen=True)
class Shard:
    """Specify how resources should be allotted for each test process"""

    #: A list of shards for each rank
    ranks: list[RankShard]

    def __str__(self) -> str:
        return "/".join(",".join(str(r) for r in rank) for rank in self.ranks)


@dataclass(frozen=True)
class StageSpec:
    """Specify the operation of a test run"""

    #: The number of worker processes to start for running tests
    workers: int

    # A list of (cpu or gpu) shardings to draw on for each test
    shards: list[Shard]


@dataclass(frozen=True)
class StageResult:
    """Collect results from all tests in a TestStage."""

    #: Individual test process results including return code and stdout.
    procs: list[ProcessResult]

    #: Cumulative execution time for all tests in a stage.
    time: timedelta

    @property
    def total(self) -> int:
        """The total number of tests run in this stage."""
        return len(self.procs)

    @property
    def passed(self) -> int:
        """The number of tests in this stage that passed."""
        return sum(p.returncode == 0 for p in self.procs)


def adjust_workers(workers: int, requested_workers: Union[int, None]) -> int:
    """Adjust computed workers according to command line requested workers.

    The final number of workers will only be adjusted down by this function.

    Parameters
    ----------
    workers: int
        The computed number of workers to use

    requested_workers: int | None, optional
        Requested number of workers from the user, if supplied (default: None)

    Returns
    -------
    int
        The number of workers to actually use

    """
    if requested_workers is not None and requested_workers < 0:
        raise ValueError("requested workers must be non-negative")

    if requested_workers is not None:
        if requested_workers > workers:
            raise RuntimeError(
                "Requested workers greater than assignable workers"
            )
        workers = requested_workers

    if workers == 0:
        raise RuntimeError("Current configuration results in zero workers")

    return workers


def log_proc(
    name: str, proc: ProcessResult, config: Config, *, verbose: bool
) -> None:
    """Log a process result according to the current configuration"""
    if config.debug or config.dry_run:
        LOG(shell(proc.invocation))
    duration = (
        f" {{{proc.time.total_seconds():0.2f}s}}"
        if proc.time is not None
        else ""
    )
    msg = f"({name}){duration} {proc.test_file}"
    details = proc.output.split("\n") if verbose else None
    if proc.skipped:
        LOG(skipped(msg))
    elif proc.timeout:
        LOG(timeout(msg))
    elif proc.returncode == 0:
        LOG(passed(msg, details=details))
    else:
        LOG(failed(msg, details=details, exit_code=proc.returncode))
