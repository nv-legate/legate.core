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

import pytest

from legate.tester.config import Config
from legate.tester.stages._linux import omp as m
from legate.tester.stages.util import UNPIN_ENV, Shard

from .. import FakeSystem

unpin_and_test = dict(UNPIN_ENV)


def test_default() -> None:
    c = Config([])
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" in stage.shard_args(Shard([shard]), c)


def test_cpu_pin_strict() -> None:
    c = Config(["test.py", "--cpu-pin", "strict"])
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.env(c, s) == {}
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" in stage.shard_args(Shard([shard]), c)


def test_cpu_pin_none() -> None:
    c = Config(["test.py", "--cpu-pin", "none"])
    s = FakeSystem(cpus=12)
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == []
    assert stage.env(c, s) == unpin_and_test
    assert stage.spec.workers > 0

    shard = (1, 2, 3)
    assert "--cpu-bind" not in stage.shard_args(Shard([shard]), c)


class TestSingleRank:
    @pytest.mark.parametrize(
        "shard,expected", [[(2,), "2"], [(1, 2, 3), "1,2,3"]]
    )
    def test_shard_args(self, shard: tuple[int, ...], expected: str) -> None:
        c = Config([])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--omps",
            f"{c.omps}",
            "--ompthreads",
            f"{c.ompthreads}",
            "--numamem",
            f"{c.numamem}",
            "--cpu-bind",
            expected,
        ]

    def test_spec_with_omps_1_threads_1(self) -> None:
        c = Config(["test.py", "--omps", "1", "--ompthreads", "1"])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 6
        assert stage.spec.shards == [
            Shard([(0, 1)]),
            Shard([(2, 3)]),
            Shard([(4, 5)]),
            Shard([(6, 7)]),
            Shard([(8, 9)]),
            Shard([(10, 11)]),
        ]

    def test_spec_with_omps_1_threads_2(self) -> None:
        c = Config(["test.py", "--omps", "1", "--ompthreads", "2"])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 4
        assert stage.spec.shards == [
            Shard([(0, 1, 2)]),
            Shard([(3, 4, 5)]),
            Shard([(6, 7, 8)]),
            Shard([(9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_1(self) -> None:
        c = Config(["test.py", "--omps", "2", "--ompthreads", "1"])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 4
        assert stage.spec.shards == [
            Shard([(0, 1, 2)]),
            Shard([(3, 4, 5)]),
            Shard([(6, 7, 8)]),
            Shard([(9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_2(self) -> None:
        c = Config(["test.py", "--omps", "2", "--ompthreads", "2"])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1, 2, 3, 4)]),
            Shard([(5, 6, 7, 8, 9)]),
        ]

    def test_spec_with_utility(self) -> None:
        c = Config(
            ["test.py", "--omps", "2", "--ompthreads", "2", "--utility", "3"]
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [Shard([(0, 1, 2, 3, 4, 5, 6)])]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(["test.py", "--omps", "1", "--ompthreads", "1", "-j", "2"])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [Shard([(0, 1)]), Shard([(2, 3)])]

    def test_spec_with_requested_workers_zero(self) -> None:
        s = FakeSystem(cpus=12)
        c = Config(["test.py", "-j", "0"])
        assert c.requested_workers == 0
        with pytest.raises(RuntimeError):
            m.OMP(c, s)

    def test_spec_with_requested_workers_bad(self) -> None:
        s = FakeSystem(cpus=12)
        c = Config(["test.py", "-j", f"{len(s.cpus)+1}"])
        assert c.requested_workers > len(s.cpus)
        with pytest.raises(RuntimeError):
            m.OMP(c, s)

    def test_spec_with_verbose(self) -> None:
        args = ["test.py", "--cpus", "2"]
        c = Config(args)
        cv = Config(args + ["--verbose"])
        s = FakeSystem(cpus=12)

        spec, vspec = m.OMP(c, s).spec, m.OMP(cv, s).spec
        assert vspec == spec


class TestMultiRank:
    @pytest.mark.parametrize(
        "shard,expected", [[(2,), "2"], [(1, 2, 3), "1,2,3"]]
    )
    def test_shard_args(self, shard: tuple[int, ...], expected: str) -> None:
        c = Config([])
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        result = stage.shard_args(Shard([shard]), c)
        assert result == [
            "--omps",
            f"{c.omps}",
            "--ompthreads",
            f"{c.ompthreads}",
            "--numamem",
            f"{c.numamem}",
            "--cpu-bind",
            expected,
        ]

    def test_spec_with_omps_1_threads_1(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "1",
                "--ompthreads",
                "1",
                "--ranks-per-node",
                "2",
            ]
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 3
        assert stage.spec.shards == [
            Shard([(0, 1), (2, 3)]),
            Shard([(4, 5), (6, 7)]),
            Shard([(8, 9), (10, 11)]),
        ]

    def test_spec_with_omps_1_threads_2(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "1",
                "--ompthreads",
                "2",
                "--ranks-per-node",
                "2",
            ]
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1, 2), (3, 4, 5)]),
            Shard([(6, 7, 8), (9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_1(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "2",
                "--ompthreads",
                "1",
                "--ranks-per-node",
                "2",
            ]
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1, 2), (3, 4, 5)]),
            Shard([(6, 7, 8), (9, 10, 11)]),
        ]

    def test_spec_with_omps_2_threads_2(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "2",
                "--ompthreads",
                "2",
                "--ranks-per-node",
                "2",
            ]
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [
            Shard([(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)]),
        ]

    def test_spec_with_utility(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "2",
                "--ompthreads",
                "2",
                "--utility",
                "3",
                "--ranks-per-node",
                "2",
            ]
        )
        s = FakeSystem(cpus=16)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 1
        assert stage.spec.shards == [
            Shard([(0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13)])
        ]

    def test_spec_with_requested_workers(self) -> None:
        c = Config(
            [
                "test.py",
                "--omps",
                "1",
                "--ompthreads",
                "1",
                "-j",
                "2",
                "--ranks-per-node",
                "2",
            ]
        )
        s = FakeSystem(cpus=12)
        stage = m.OMP(c, s)
        assert stage.spec.workers == 2
        assert stage.spec.shards == [
            Shard([(0, 1), (2, 3)]),
            Shard([(4, 5), (6, 7)]),
        ]
