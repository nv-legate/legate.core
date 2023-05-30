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

import os
import platform

import pytest
from pytest_mock import MockerFixture

import legate.util.system as m


def test___all__() -> None:
    assert m.__all__ == ("System",)


class TestSystem:
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("____TEST", "10")

        s = m.System()

        expected = dict(os.environ)
        expected.update({"____TEST": "10"})
        assert s.env == expected

        assert id(s.env) != id(os.environ)

    @pytest.mark.parametrize("os", ("Linux", "Darwin"))
    def test_os_good(self, mocker: MockerFixture, os: str) -> None:
        mocker.patch("platform.system", return_value=os)

        s = m.System()

        assert s.os == os

    def test_os_bad(self, mocker: MockerFixture) -> None:
        mocker.patch("platform.system", return_value="junk")

        s = m.System()

        msg = "Legate does not work on junk"
        with pytest.raises(RuntimeError, match=msg):
            s.os

    def test_LIBPATH_Linux(self, mocker: MockerFixture) -> None:
        mocker.patch("platform.system", return_value="Linux")

        s = m.System()

        assert s.LIB_PATH == "LD_LIBRARY_PATH"

    def test_LIBPATH_Darwin(self, mocker: MockerFixture) -> None:
        mocker.patch("platform.system", return_value="Darwin")

        s = m.System()

        assert s.LIB_PATH == "DYLD_LIBRARY_PATH"

    # These properties delegate to util functions, just verify plumbing

    def test_legate_paths(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "legate.util.system.get_legate_paths",
            return_value="legate paths",
        )

        s = m.System()

        assert s.legate_paths == "legate paths"  # type: ignore

    def test_legion_paths(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "legate.util.system.get_legion_paths",
            return_value="legion paths",
        )

        s = m.System()

        assert s.legion_paths == "legion paths"  # type: ignore

    def test_cpus(self) -> None:
        s = m.System()
        cpus = s.cpus
        assert len(cpus) > 0
        assert all(len(cpu.ids) > 0 for cpu in cpus)

    @pytest.mark.skipif(platform.system() != "Darwin", reason="OSX test")
    def test_gpus_osx(self) -> None:
        s = m.System()

        msg = "GPU execution is not available on OSX."
        with pytest.raises(RuntimeError, match=msg):
            s.gpus
