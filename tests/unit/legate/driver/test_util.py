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
from subprocess import CalledProcessError

import pytest

import legate.driver.util as m


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


HEADER_PATH = Path(__file__).parent / "sample_header.h"


def test_read_c_define_hit() -> None:
    assert m.read_c_define(HEADER_PATH, "FOO") == "10"
    assert m.read_c_define(HEADER_PATH, "BAR") == '"bar"'


def test_read_c_define_miss() -> None:
    assert m.read_c_define(HEADER_PATH, "JUNK") is None


CMAKE_CACHE_PATH = Path(__file__).parent / "sample_cmake_cache.txt"


def test_grep_file_hit() -> None:
    assert (
        m.grep_file("Legion_SOURCE_DIR:STATIC=", CMAKE_CACHE_PATH)
        == '"foo/bar"'
    )
    assert (
        m.grep_file("FIND_LEGATE_CORE_CPP:BOOL=OFF", CMAKE_CACHE_PATH) == "OFF"
    )


def test_grep_file_miss() -> None:
    with pytest.raises(CalledProcessError):
        assert m.grep_file("JUNK", CMAKE_CACHE_PATH) is None
