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

import multiprocessing
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from typing_extensions import Protocol

from ...util.colors import yellow
from ...util.types import ArgList, EnvDict
from ...util.ui import banner, summary
from .. import CUSTOM_FILES, PER_FILE_ARGS, FeatureType
from ..config import Config
from ..test_system import ProcessResult, TestSystem
from .util import Shard, StageResult, StageSpec, log_proc


class TestStage(Protocol):
    """Encapsulate running configured test files using specific features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType

    #: The computed specification for processes to launch to run the
    #: configured test files.
    spec: StageSpec

    #: The computed sharding id sets to use for job runs
    shards: queue.Queue[Any]

    #: After the stage completes, results will be stored here
    result: StageResult

    #: Any fixed stage-specific command-line args to pass
    args: ArgList

    #: Shard assigned to each worker in the pool.
    #: Each worker in the pool will get a local copy of this attribute,
    #: will populate it on first touch by pulling from the global shard
    #: queue, then keep using its assigned shard throughout the run.
    worker_shard: Union[Shard, None] = None

    # --- Protocol methods

    def __init__(self, config: Config, system: TestSystem) -> None:
        ...

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        """Generate stage-specific customizations to the process env

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        ...

    def delay(self, shard: Shard, config: Config, system: TestSystem) -> None:
        """Wait any delay that should be applied before running the next
        test.

        Parameters
        ----------
        shard: Shard
            The shard to be used for the next test that is run

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        return

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        """Generate the command line arguments necessary to launch
        the next test process on the given shard.

        Parameters
        ----------
        shard: Shard
            The shard to be used for the next test that is run

        config: Config
            Test runner configuration

        """
        ...

    def compute_spec(self, config: Config, system: TestSystem) -> StageSpec:
        """Compute the number of worker processes to launch and stage shards
        to use for running the configured test files.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        ...

    # --- Shared implementation methods

    def __call__(self, config: Config, system: TestSystem) -> None:
        """Execute this test stage.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        t0 = datetime.now()
        procs = self._launch(config, system)
        t1 = datetime.now()

        self.result = StageResult(procs, t1 - t0)

    @property
    def name(self) -> str:
        """A stage name to display for tests in this stage."""
        return self.__class__.__name__

    @property
    def intro(self) -> str:
        """An informative banner to display at stage end."""
        workers = self.spec.workers
        workers_text = f"{workers} worker{'s' if workers > 1 else ''}"
        return (
            banner(f"Entering stage: {self.name} (with {workers_text})") + "\n"
        )

    @property
    def outro(self) -> str:
        """An informative banner to display at stage end."""
        total, passed = self.result.total, self.result.passed

        result = summary(self.name, total, passed, self.result.time)

        footer = banner(
            f"Exiting stage: {self.name}",
            details=(
                "* Results      : "
                + yellow(
                    f"{passed} / {total} files passed "  # noqa E500
                    f"({passed/total*100:0.1f}%)"
                    if total > 0
                    else "0 tests are running, Please check "
                ),
                "* Elapsed time : " + yellow(f"{self.result.time}"),
            ),
        )

        return f"{result}\n{footer}"

    def file_args(self, test_file: Path, config: Config) -> ArgList:
        """Extra command line arguments based on the test file.

        Parameters
        ----------
        test_file : Path
            Path to a test file

        config: Config
            Test runner configuration

        """
        test_file_string = str(test_file)
        args = PER_FILE_ARGS.get(test_file_string, [])

        # These are a bit ugly but necessary in order to make pytest generate
        # more verbose output for integration tests when -v, -vv is specified
        if "integration" in test_file_string and config.verbose > 0:
            args += ["-v"]
        if "integration" in test_file_string and config.verbose > 1:
            args += ["-s"]

        return args

    def cov_args(self, config: Config) -> ArgList:
        """Coverage binary and coverage arguments.

        Parameters
        ----------
        config: Config
            Test runner configuration

        """
        if config.cov_bin:
            args = [str(config.cov_bin)] + config.cov_args.split()
            if config.cov_src_path:
                args += ["--source", str(config.cov_src_path)]
        else:
            args = []

        return args

    def run(
        self,
        test_file: Path,
        config: Config,
        system: TestSystem,
        *,
        custom_args: ArgList | None = None,
    ) -> ProcessResult:
        """Execute a single test files with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_file : Path
            Test file to execute

        config: Config
            Test runner configuration

        system: TestSystem
            Process execution wrapper

        """
        if self.worker_shard is None:
            self.worker_shard = self.shards.get()

        test_path = config.root_dir / test_file

        cov_args = self.cov_args(config)

        stage_args = self.args + self.shard_args(self.worker_shard, config)
        file_args = self.file_args(test_file, config)

        cmd = (
            [str(config.legate_path)]
            + stage_args
            + cov_args
            + [str(test_path)]
            + file_args
            + config.extra_args
        )

        if custom_args:
            cmd += custom_args

        self.delay(self.worker_shard, config, system)

        result = system.run(
            cmd,
            test_file,
            env=self._env(config, system),
            timeout=config.timeout,
        )
        log_proc(self.name, result, config, verbose=config.verbose)

        return result

    def _env(self, config: Config, system: TestSystem) -> EnvDict:
        env = dict(config.env)
        env.update(self.env(config, system))

        # special case for LEGATE_CONFIG -- if users have specified this on
        # their own we still want to see the value since it will affect the
        # test invocation directly.
        if "LEGATE_CONFIG" in system.env:
            env["LEGATE_CONFIG"] = system.env["LEGATE_CONFIG"]

        return env

    def _init(self, config: Config, system: TestSystem) -> None:
        self.spec = self.compute_spec(config, system)
        self.shards = system.manager.Queue(len(self.spec.shards))
        for shard in self.spec.shards:
            self.shards.put(shard)

    def _launch(
        self, config: Config, system: TestSystem
    ) -> list[ProcessResult]:
        pool = multiprocessing.pool.ThreadPool(self.spec.workers)

        jobs = [
            pool.apply_async(self.run, (path, config, system))
            for path in config.test_files
        ]
        pool.close()

        sharded_results = [job.get() for job in jobs]

        custom = (x for x in CUSTOM_FILES if x.kind == self.kind)

        custom_results = [
            self.run(Path(x.file), config, system, custom_args=x.args)
            for x in custom
        ]

        return sharded_results + custom_results
