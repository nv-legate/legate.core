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
from itertools import chain

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

class Test_expand_range:

    def test_errors(self) -> None:
        msg = ""
        with pytest.raises(ValueError):
            m.expand_range("foo")

    def test_empty(self) -> None:
        assert m.expand_range("") == ()

    @pytest.mark.parametrize("val", ("0", "1", "12", "100"))
    def test_single_number(self, val) -> None:
        assert m.expand_range(val) == (int(val),)

    @pytest.mark.parametrize("val", ("0-10", "1-2", "12-25"))
    def test_range(self, val) -> None:
        start, stop = val.split('-')
        assert m.expand_range(val) == tuple(range(int(start), int(stop) + 1))


class Test_extract_values:

    def test_errors(self) -> None:
        msg = ""
        with pytest.raises(ValueError):
            m.extract_values("foo")

    def test_empty(self) -> None:
        assert m.extract_values("") == ()

    @pytest.mark.parametrize("val", ("0", "1,2", "3,5,7"))
    def test_individual(self, val) -> None:
        expected = {"0": (0,),
                    "1,2": (1, 2),
                    "3,5,7": (3, 5, 7),
                    }
        assert m.extract_values(val) == expected[val]

    @pytest.mark.parametrize("val", ("2,1", "8,5,3,2", "1,3,2,5,4,7,6"))
    def test_individual_ordered(self, val) -> None:
        expected = {"2,1": (1, 2),
                    "8,5,3,2": (2, 3, 5, 8),
                    "1,3,2,5,4,7,6": (1, 2, 3, 4, 5, 6, 7),
                    }
        assert m.extract_values(val) == expected[val]

    @pytest.mark.parametrize("val", ("0-2", "0-2,4-5", "0-1,3-5,8-11"))
    def test_range(self, val) -> None:
        expected = {"0-2": (0, 1, 2),
                    "0-2,4-5": (0, 1, 2, 4, 5),
                    "0-1,3-5,8-11": (0, 1, 3, 4, 5, 8, 9, 10 ,11),
                    }
        assert m.extract_values(val) == expected[val]

    @pytest.mark.parametrize("val", ("2-3,0-1", "0-1,4-5,2-3"))
    def test_range_ordered(self, val) -> None:
        expected = {"2-3,0-1": (0, 1, 2, 3),
                    "0-1,4-5,2-3": (0, 1, 2, 3, 4, 5),
                    }
        assert m.extract_values(val) == expected[val]

    @pytest.mark.parametrize("val", ("0,1-2", "1-2,0", "0,1-2,3,4-5,6", "5-6,4,1-3,0"))
    def test_mixed(self, val) -> None:
        expected = {"0,1-2": (0, 1, 2),
                    "1-2,0": (0, 1, 2),
                    "0,1-2,3,4-5,6": (0, 1, 2, 3, 4, 5, 6),
                    "5-6,4,1-3,0": (0, 1, 2, 3, 4, 5, 6),
                    }
        assert m.extract_values(val) == expected[val]
