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
"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

from ..util import colors
from ..util.types import ArgList, EnvDict
from . import (
    DEFAULT_PROCESS_ENV,
    FEATURES,
    LAST_FAILED_FILENAME,
    SKIPPED_EXAMPLES,
    FeatureType,
)
from .args import parser


class Config:
    """A centralized configuration object that provides the information
    needed by test stages in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList) -> None:
        self.argv = argv

        args, self._extra_args = parser.parse_known_args(self.argv[1:])

        colors.ENABLED = args.color

        # which tests to run
        self.examples = False if args.cov_bin else True
        self.integration = True
        self.unit = args.unit
        self.files = args.files
        self.last_failed = args.last_failed

        # feature configuration
        self.features = self._compute_features(args)

        # feature options for integration tests
        self.cpus = args.cpus
        self.gpus = args.gpus
        self.omps = args.omps
        self.utility = args.utility
        self.cpu_pin = args.cpu_pin
        self.fbmem = args.fbmem
        self.bloat_factor = args.bloat_factor
        self.gpu_delay = args.gpu_delay
        self.ompthreads = args.ompthreads
        self.numamem = args.numamem
        self.ranks = args.ranks

        # test run configuration
        self.timeout = args.timeout
        self.debug = args.debug
        self.dry_run = args.dry_run
        self.verbose = args.verbose
        self.test_root = args.test_root
        self.requested_workers = args.workers
        self.legate_dir = self._compute_legate_dir(args)
        self.cov_bin = args.cov_bin
        self.cov_args = args.cov_args
        self.cov_src_path = args.cov_src_path

    @property
    def env(self) -> EnvDict:
        """Custom environment settings used for process exectution."""
        return dict(DEFAULT_PROCESS_ENV)

    @property
    def extra_args(self) -> ArgList:
        """Extra command-line arguments to pass on to individual test files."""
        return self._extra_args

    @property
    def root_dir(self) -> Path:
        """Path to the directory containing the tests."""
        if self.test_root:
            return Path(self.test_root)

        # if not explicitly given, just use cwd assuming we are at a repo top
        return Path(os.getcwd())

    @property
    def test_files(self) -> tuple[Path, ...]:
        """List of all test files to use for each stage.

        An explicit list of files from the command line will take precedence.

        Otherwise, the files are computed based on command-line options, etc.

        """
        if self.files and self.last_failed:
            raise RuntimeError("Cannot specify both --files and --last-failed")

        if self.files:
            return self.files

        if self.last_failed:
            if last_failed := self._read_last_failed():
                return last_failed

        files = []

        if self.examples:
            examples = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("examples").glob("*.py")
                if str(path.relative_to(self.root_dir)) not in SKIPPED_EXAMPLES
            )
            files.extend(sorted(examples))

        if self.integration:
            integration_tests = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("tests/integration").glob(
                    "*.py"
                )
            )
            files.extend(sorted(integration_tests))

        if self.unit:
            unit_tests = (
                path.relative_to(self.root_dir)
                for path in self.root_dir.joinpath("tests/unit").glob(
                    "**/*.py"
                )
            )
            files.extend(sorted(unit_tests))

        return tuple(files)

    @property
    def legate_path(self) -> str:
        """Computed path to the legate driver script"""
        if self.legate_dir is None:
            return "legate"
        return str(self.legate_dir / "bin" / "legate")

    def _compute_features(self, args: Namespace) -> tuple[FeatureType, ...]:
        if args.features is not None:
            computed = args.features
        else:
            computed = [
                feature
                for feature in FEATURES
                if os.environ.get(f"USE_{feature.upper()}", None) == "1"
            ]

        # if nothing is specified any other way, at least run CPU stage
        if len(computed) == 0:
            computed.append("cpus")

        return tuple(computed)

    def _compute_legate_dir(self, args: Namespace) -> Path | None:
        # self._legate_source below is purely for testing
        if args.legate_dir:
            self._legate_source = "cmd"
            return Path(args.legate_dir)
        elif "LEGATE_DIR" in os.environ:
            self._legate_source = "env"
            return Path(os.environ["LEGATE_DIR"])
        self._legate_source = "install"
        return None

    def _read_last_failed(self) -> tuple[Path, ...]:
        try:
            with open(LAST_FAILED_FILENAME) as f:
                lines = (line for line in f.readlines() if line.strip())
                return tuple(Path(line.strip()) for line in lines)
        except OSError:
            return ()
