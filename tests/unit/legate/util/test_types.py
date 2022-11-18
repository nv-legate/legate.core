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

from dataclasses import dataclass

import legate.util.types as m


class TestCPUInfo:
    def test_fields(self) -> None:
        assert set(m.CPUInfo.__dataclass_fields__) == {"ids"}


class TestGPUInfo:
    def test_fields(self) -> None:
        assert set(m.GPUInfo.__dataclass_fields__) == {"id", "total"}


class Source:
    foo = 10
    bar = 10.2
    baz = "test"
    quux = ["a", "b", "c"]
    extra = (1, 2, 3)


@dataclass(frozen=True)
class Target(m.DataclassMixin):
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
