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

from dataclasses import dataclass

import legate.driver.types as m
from legate.driver.ui import scrub


def test___all__() -> None:
    assert m.__all__ == (
        "ArgList",
        "Command",
        "CommandPart",
        "DataclassMixin",
        "DataclassProtocol",
        "EnvDict",
        "LauncherType",
        "LegatePaths",
        "LegionPaths",
    )


class TestDataClassMixin:
    def test_str(self) -> None:
        @dataclass(frozen=True)
        class Foo(m.DataclassMixin):
            bar: str
            baz: float
            quux: int

        obj = Foo("bar", 123.456, 10)

        assert scrub(str(obj)) == "bar  : bar\nbaz  : 123.456\nquux : 10"


class TestLegatePaths:
    def test_fields(self) -> None:
        assert set(m.LegatePaths.__dataclass_fields__) == {
            "legate_build_dir",
            "bind_sh_path",
            "legate_dir",
            "legate_lib_path",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.LegatePaths, m.DataclassMixin)


class TestLegionPaths:
    def test_fields(self) -> None:
        assert set(m.LegionPaths.__dataclass_fields__) == {
            "realm_defines_h",
            "legion_python",
            "legion_bin_path",
            "legion_spy_py",
            "legion_module",
            "legion_prof_py",
            "legion_lib_path",
            "legion_defines_h",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.LegionPaths, m.DataclassMixin)
