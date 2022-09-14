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
from pathlib import Path
from shlex import quote

import pytest
from util import Capsys

import legate.driver.util as m
from legate.driver.config import Config
from legate.driver.driver import Driver
from legate.driver.system import System
from legate.driver.ui import scrub


class Source:
    foo = 10
    bar = 10.2
    baz = "test"
    quux = ["a", "b", "c"]
    extra = (1, 2, 3)


@dataclass(frozen=True)
class Target:
    foo: int
    bar: float
    baz: str
    quux: list[str]


def test_object_to_dataclass() -> None:
    source = Source()
    target = m.object_to_dataclass(source, Target)

    assert set(target.__dict__) == set(Target.__dataclass_fields__)
    for k, v in target.__dict__.items():
        assert getattr(source, k) == v


class Test_print_verbose:
    def test_system_only(self, capsys: Capsys) -> None:
        system = System()

        m.print_verbose(system)

        out = scrub(capsys.readouterr()[0]).strip()

        assert out.startswith(f"{'--- Legion Python Configuration ':-<80}")
        assert "Legate paths:" in out
        for line in scrub(str(system.legate_paths)).split():
            assert line in out

        assert "Legion paths:" in out
        for line in scrub(str(system.legion_paths)).split():
            assert line in out

    def test_system_and_driver(self, capsys: Capsys) -> None:
        config = Config(["legate", "--no-replicate"])
        system = System()
        driver = Driver(config, system)

        m.print_verbose(system, driver)

        out = scrub(capsys.readouterr()[0]).strip()

        assert out.startswith(f"{'--- Legion Python Configuration ':-<80}")
        assert "Legate paths:" in out
        for line in scrub(str(system.legate_paths)).split():
            assert line in out

        assert "Legion paths:" in out
        for line in scrub(str(system.legion_paths)).split():
            assert line in out

        assert "Command:" in out
        assert f"  {' '.join(quote(t) for t in driver.cmd)}" in out

        assert "Customized Environment:" in out
        for k in driver.custom_env_vars:
            assert f"{k}={driver.env[k]}" in out

        assert out.endswith(f"\n{'-':-<80}")


HEADER_PATH = Path(__file__).parent / "sample_header.h"


def test_read_c_define_hit() -> None:
    assert m.read_c_define(HEADER_PATH, "FOO") == "10"
    assert m.read_c_define(HEADER_PATH, "BAR") == '"bar"'


def test_read_c_define_miss() -> None:
    assert m.read_c_define(HEADER_PATH, "JUNK") is None


CMAKE_CACHE_PATH = Path(__file__).parent / "sample_cmake_cache.txt"


def test_read_cmake_cache_value_hit() -> None:
    assert (
        m.read_cmake_cache_value(CMAKE_CACHE_PATH, "Legion_SOURCE_DIR:STATIC=")
        == '"foo/bar"'
    )
    assert (
        m.read_cmake_cache_value(
            CMAKE_CACHE_PATH, "FIND_LEGATE_CORE_CPP:BOOL=OFF"
        )
        == "OFF"
    )


def test_read_cmake_cache_value_miss() -> None:
    with pytest.raises(RuntimeError):
        assert m.read_cmake_cache_value(CMAKE_CACHE_PATH, "JUNK") is None
