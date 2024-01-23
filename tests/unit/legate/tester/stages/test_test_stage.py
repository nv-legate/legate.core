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

from datetime import timedelta
from pathlib import Path

from legate.tester import FeatureType
from legate.tester.config import Config
from legate.tester.stages import test_stage as m
from legate.tester.stages.util import Shard, StageResult, StageSpec
from legate.tester.test_system import ProcessResult, TestSystem as _TestSystem
from legate.util.types import ArgList, EnvDict

from . import FakeSystem


class MockTestStage(m.TestStage):
    kind: FeatureType = "eager"

    name = "mock"

    args = ["-foo", "-bar"]

    def __init__(self, config: Config, system: _TestSystem) -> None:
        self._init(config, system)

    def compute_spec(self, config: Config, system: _TestSystem) -> StageSpec:
        shards = [Shard([(0,)]), Shard([(1,)]), Shard([(2,)])]
        return StageSpec(2, shards)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return []

    def env(self, config: Config, system: _TestSystem) -> EnvDict:
        return {}


class TestTestStage:
    def test_name(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert stage.name == "mock"

    def test_intro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert "Entering stage: mock" in stage.intro

    def test_outro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        stage.result = StageResult(
            [ProcessResult("invoke", Path("test/file"))],
            timedelta(seconds=2.123),
        )
        outro = stage.outro
        assert "Exiting stage: mock" in outro
        assert "Passed 1 of 1 tests (100.0%)" in outro
        assert "2.123" in outro

    def test_file_args_default(self) -> None:
        c = Config([])
        stage = MockTestStage(c, FakeSystem())
        assert stage.file_args(Path("integration/foo"), c) == []
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_file_args_v(self) -> None:
        c = Config(["test.py", "-v"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.file_args(Path("integration/foo"), c) == ["-v"]
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_file_args_vv(self) -> None:
        c = Config(["test.py", "-vv"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.file_args(Path("integration/foo"), c) == ["-v", "-s"]
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_cov_args_without_cov_bin(self) -> None:
        c = m.Config(["test.py", "--cov-args", "run -a"])
        stage = MockTestStage(c, FakeSystem())
        assert stage.cov_args(c) == []

    def test_cov_args_with_cov_bin(self) -> None:
        cov_bin = "conda/envs/legate/bin/coverage"
        args = ["--cov-bin", cov_bin]
        c = m.Config(["test.py"] + args)
        expected_result = [cov_bin] + c.cov_args.split()
        stage = MockTestStage(c, FakeSystem())
        assert stage.cov_args(c) == expected_result

    def test_cov_args_with_cov_bin_args_and_src_path(self) -> None:
        cov_bin = "conda/envs/legate/bin/coverage"
        cov_args = "run -a"
        cov_src_path = "source_path"
        args = (
            ["--cov-bin", cov_bin]
            + ["--cov-args", cov_args]
            + ["--cov-src-path", cov_src_path]
        )
        c = m.Config(["test.py"] + args)
        expected_result = (
            [cov_bin] + cov_args.split() + ["--source", cov_src_path]
        )
        stage = MockTestStage(c, FakeSystem())
        assert stage.cov_args(c) == expected_result
