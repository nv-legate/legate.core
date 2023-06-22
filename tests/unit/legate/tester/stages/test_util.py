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

import pytest

from legate.tester.config import Config
from legate.tester.logger import LOG
from legate.tester.stages import util as m
from legate.tester.test_system import ProcessResult
from legate.util.ui import failed, passed, shell, skipped, timeout


def test_StageResult() -> None:
    procs = [ProcessResult(f"run{i}", Path(f"test{i}")) for i in range(10)]
    procs[2].returncode = 10
    procs[7].returncode = -2

    result = m.StageResult(procs=procs, time=timedelta(0))

    assert result.total == 10
    assert result.passed == 8


class Test_adjust_workers:
    @pytest.mark.parametrize("n", (1, 5, 100))
    def test_None_requested(self, n: int) -> None:
        assert m.adjust_workers(n, None) == n

    @pytest.mark.parametrize("n", (1, 2, 9))
    def test_requested(self, n: int) -> None:
        assert m.adjust_workers(10, n) == n

    def test_negative_requested(self) -> None:
        with pytest.raises(ValueError):
            assert m.adjust_workers(10, -1)

    def test_zero_requested(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(10, 0)

    def test_zero_computed(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(0, None)

    def test_requested_too_large(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(10, 11)


class Test_log_proc:
    @pytest.mark.parametrize("returncode", (-23, -1, 0, 1, 17))
    def test_skipped(self, returncode: int) -> None:
        config = Config([])
        proc = ProcessResult(
            "proc", Path("proc"), skipped=True, returncode=returncode
        )

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (skipped(f"(foo) {proc.test_file}"),)

    def test_passed(self) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"))

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (passed(f"(foo) {proc.test_file}"),)

    def test_passed_verbose(self) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"), output="foo\nbar")
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            passed(f"(foo) {proc.test_file}", details=details).split("\n")
        )

    @pytest.mark.parametrize("returncode", (-23, -1, 1, 17))
    def test_failed(self, returncode: int) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"), returncode=returncode)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            failed(f"(foo) {proc.test_file}", exit_code=returncode),
        )

    @pytest.mark.parametrize("returncode", (-23, -1, 1, 17))
    def test_failed_verbose(self, returncode: int) -> None:
        config = Config([])
        proc = ProcessResult(
            "proc", Path("proc"), returncode=returncode, output="foo\nbar"
        )
        details = proc.output.split("\n")

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=True)

        assert LOG.lines == tuple(
            failed(
                f"(foo) {proc.test_file}",
                details=details,
                exit_code=returncode,
            ).split("\n")
        )

    def test_timeout(self) -> None:
        config = Config([])
        proc = ProcessResult("proc", Path("proc"), timeout=True)

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (timeout(f"(foo) {proc.test_file}"),)

    def test_dry_run(self) -> None:
        config = Config([])
        config.dry_run = True
        proc = ProcessResult("proc", Path("proc"))

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            shell(proc.invocation),
            passed(f"(foo) {proc.test_file}"),
        )

    def test_debug(self) -> None:
        config = Config([])
        config.debug = True
        proc = ProcessResult("proc", Path("proc"))

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            shell(proc.invocation),
            passed(f"(foo) {proc.test_file}"),
        )

    def test_time(self) -> None:
        config = Config([])
        config.debug = True
        proc = ProcessResult(
            "proc", Path("proc"), time=timedelta(seconds=2.41)
        )

        LOG.clear()
        m.log_proc("foo", proc, config, verbose=False)

        assert LOG.lines == (
            shell(proc.invocation),
            passed(f"(foo) {{2.41s}} {proc.test_file}"),
        )
