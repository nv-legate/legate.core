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
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from legate.tester import test_system as m


class TestProcessResult:
    def test_default(self) -> None:
        ret = m.ProcessResult("proc", Path("proc"))
        assert ret.invocation == "proc"
        assert ret.test_file == Path("proc")
        assert ret.time is None
        assert not ret.skipped
        assert not ret.timeout
        assert ret.returncode == 0
        assert ret.output == ""
        assert ret.passed

    def test_passed_skipped(self) -> None:
        ret = m.ProcessResult("proc", Path("proc"), skipped=True)
        assert ret.passed

    def test_passed_return_zero(self) -> None:
        ret = m.ProcessResult("proc", Path("proc"), returncode=0)
        assert ret.passed

    def test_passed_return_nonzero(self) -> None:
        ret = m.ProcessResult("proc", Path("proc"), returncode=1)
        assert not ret.passed

    def test_passed_timeout(self) -> None:
        ret = m.ProcessResult("proc", Path("proc"), timeout=True)
        assert not ret.passed


@pytest.fixture
def mock_subprocess_run(mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(m, "stdlib_run")


CMD = "legate script.py --cpus 4"


class TestSystem:
    def test_init(self) -> None:
        s = m.TestSystem()
        assert s.dry_run is False

    def test_run(self, mock_subprocess_run: MagicMock) -> None:
        s = m.TestSystem()

        mock_subprocess_run.return_value = CompletedProcess(
            CMD, 10, stdout="<output>"
        )

        result = s.run(CMD.split(), Path("test/file"))
        mock_subprocess_run.assert_called()

        assert result.invocation == CMD
        assert result.test_file == Path("test/file")
        assert result.time is not None and result.time > timedelta(0)
        assert not result.skipped
        assert not result.timeout
        assert result.returncode == 10
        assert result.output == "<output>"
        assert not result.passed

    def test_dry_run(self, mock_subprocess_run: MagicMock) -> None:
        s = m.TestSystem(dry_run=True)

        result = s.run(CMD.split(), Path("test/file"))
        mock_subprocess_run.assert_not_called()

        assert result.output == ""
        assert result.skipped

    def test_timeout(self) -> None:
        s = m.TestSystem()

        result = s.run(["sleep", "2"], Path("test/file"), timeout=1)

        assert result.timeout
        assert not result.skipped
