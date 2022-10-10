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

from pathlib import Path

import pytest

import legate.util.fs as m

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
