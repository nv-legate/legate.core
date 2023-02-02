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
from __future__ import annotations

import struct

import pytest

from legate.core import BufferBuilder
from legate.core.machine import (
    Machine,
    ProcessorKind,
    ProcessorRange,
    sanitize_kind,
)


class TestProcessorKind:
    def test_names(self) -> None:
        assert set(k.name for k in ProcessorKind) == {"GPU", "OMP", "CPU"}

    @pytest.mark.parametrize("kind", ["GPU", "OMp", "cpu"])
    def test_sanitize_str(self, kind: str) -> None:
        assert sanitize_kind(kind).name == kind.upper()

    @pytest.mark.parametrize("kind", set(ProcessorKind))
    def test_sanitize_enum(self, kind: ProcessorKind) -> None:
        assert sanitize_kind(kind) is kind

    def test_sanitize_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid processor kind: tpu"):
            sanitize_kind("tpu")


class TestProcessorRange:
    def test_create_nonempty(self) -> None:
        r = ProcessorRange.create(ProcessorKind.CPU, 1, 1, 3)
        assert not r.empty
        assert r.kind == ProcessorKind.CPU
        assert r.per_node_count == 1 and r.lo == 1 and r.hi == 3
        assert len(r) == 3

        assert r.get_node_range() == (1, 3)

    def test_create_empty(self) -> None:
        r = ProcessorRange.create(ProcessorKind.GPU, 1, 1, 0)
        assert r.empty
        assert r.kind == ProcessorKind.GPU
        assert r.per_node_count == 1 and r.lo == 1 and r.hi == 0
        assert len(r) == 0

        r = ProcessorRange.create(ProcessorKind.GPU, 1, 2, 1)
        assert r.empty
        assert r.lo == 1 and r.hi == 0
        assert len(r) == 0

        err_msg = "Illegal to get a node range of an empty processor range"
        with pytest.raises(ValueError, match=err_msg):
            r.get_node_range()

    def test_intersection_nonempty(self) -> None:
        r1 = ProcessorRange.create(ProcessorKind.GPU, 1, 0, 3)
        r2 = ProcessorRange.create(ProcessorKind.GPU, 1, 2, 4)
        r3 = r1 & r2
        assert r3 == ProcessorRange.create(ProcessorKind.GPU, 1, 2, 3)

    def test_intersection_empty(self) -> None:
        r1 = ProcessorRange.create(ProcessorKind.GPU, 1, 0, 2)
        r2 = ProcessorRange.create(ProcessorKind.GPU, 1, 3, 5)
        r3 = r1 & r2
        assert r3 == ProcessorRange.create(ProcessorKind.GPU, 1, 1, 0)
        assert len(r3) == 0

    def test_intersection_invalid(self) -> None:
        r1 = ProcessorRange.create(ProcessorKind.CPU, 1, 1, 0)
        r2 = ProcessorRange.create(ProcessorKind.GPU, 1, 1, 0)
        err_msg = (
            "Intersection between different processor kinds: " "CPU and GPU"
        )
        with pytest.raises(ValueError, match=err_msg):
            r1 & r2

    def test_empty_slice(self) -> None:
        r = ProcessorRange.create(ProcessorKind.GPU, 1, 2, 5)
        assert len(r.slice(slice(0, 0))) == 0
        assert len(r.slice(slice(None, 0))) == 0
        assert len(r.slice(slice(4, None))) == 0

        r = ProcessorRange.create(ProcessorKind.GPU, 1, 3, 1)
        assert len(r.slice(slice(0, 0))) == 0
        assert len(r.slice(slice(None, 0))) == 0
        assert len(r.slice(slice(4, None))) == 0

    def test_nonempty_slice(self) -> None:
        r = ProcessorRange.create(ProcessorKind.GPU, 1, 2, 5)
        assert len(r.slice(slice(None))) == len(r)
        for i in range(len(r)):
            assert len(r.slice(slice(i))) == i
            assert len(r.slice(slice(i, None))) == len(r) - i

        r = ProcessorRange.create(ProcessorKind.GPU, 1, 3, 1)
        assert len(r.slice(slice(None))) == 0
        for i in range(len(r)):
            assert len(r.slice(slice(i))) == 0
            assert len(r.slice(slice(i, None))) == 0

    def test_invalid_slice(self) -> None:
        r = ProcessorRange.create(ProcessorKind.GPU, 1, 2, 4)
        with pytest.raises(ValueError, match="The slicing step must be 1"):
            r.slice(slice(None, None, 2))

    def test_pack(self) -> None:
        r = ProcessorRange(ProcessorKind.CPU, 3, 4, 5)
        buf = BufferBuilder()
        r.pack(buf)
        packed = buf.get_string()
        assert packed is not None
        assert buf.get_size() == 12
        v1, v2, v3 = struct.unpack("III", packed)
        assert v1 == 3
        assert v2 == 4
        assert v3 == 5


RANGES = [
    ProcessorRange.create(ProcessorKind.CPU, 4, 1, 3),
    ProcessorRange.create(ProcessorKind.OMP, 2, 0, 3),
    ProcessorRange.create(ProcessorKind.GPU, 3, 3, 6),
]
EMPTY_RANGE = ProcessorRange.create(ProcessorKind.GPU, 1, 1, 0)


class TestMachine:
    def test_empty_machine(self) -> None:
        m = Machine([])
        assert m.preferred_kind == ProcessorKind.CPU
        assert len(m) == 0
        assert len(m.get_processor_range("cpu")) == 0
        err_msg = "Illegal to get a node range of an empty processor range"
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range()
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range("gpu")

    def test_eq(self) -> None:
        m1 = Machine(RANGES[:-1])

        m2 = Machine(RANGES[:-1])
        assert m1 == m2

        m3 = Machine([])
        assert m1 != m3

        ranges = RANGES[:-1] + [
            ProcessorRange.create(ProcessorKind.GPU, 1, 1, 0)
        ]
        m4 = Machine(ranges)
        assert m1 == m4

    @pytest.mark.parametrize("n", range(1, len(RANGES) + 1))
    def test_preferred_kind(self, n: int) -> None:
        m = Machine(RANGES[:n])
        assert m.preferred_kind == RANGES[n - 1].kind
        assert len(m) == len(RANGES[n - 1])

    def test_get_processor_range(self) -> None:
        m = Machine(RANGES[:-1])
        assert m.get_processor_range(ProcessorKind.CPU) == RANGES[0]
        assert m.get_processor_range(ProcessorKind.OMP) == RANGES[1]
        assert m.get_processor_range("cpu") == RANGES[0]
        assert m.get_processor_range("omp") == RANGES[1]
        assert m.get_processor_range() == RANGES[1]
        assert len(m.get_processor_range(ProcessorKind.GPU)) == 0
        assert len(m.get_processor_range("gpu")) == 0

    def test_get_node_range(self) -> None:
        m = Machine(RANGES)
        assert m.get_node_range("cpu") == (0, 0)
        assert m.get_node_range("omp") == (0, 1)
        assert m.get_node_range("gpu") == (1, 2)
        assert m.get_node_range() == m.get_node_range("gpu")

    def test_only_remove(self) -> None:
        m = Machine(RANGES)
        assert len(m.only("gpu")) == len(RANGES[-1])
        assert len(m.remove("gpu")) == len(RANGES[-2])
        assert m.only("gpu").only("gpu") == m.only("gpu")
        assert m.remove("gpu").remove("gpu") == m.remove("gpu")
        assert len(m.only("gpu").get_processor_range("cpu")) == 0
        assert m.only("gpu").get_processor_range("gpu") == RANGES[-1]
        assert len(m.remove("gpu").get_processor_range("gpu")) == 0
        assert m.remove("gpu").get_processor_range("omp") == RANGES[1]

    def test_count(self) -> None:
        m = Machine(RANGES)
        assert m.count("cpu") == len(RANGES[0])
        assert m.count("omp") == len(RANGES[1])
        assert m.count("gpu") == len(RANGES[2])

    def test_get_item(self) -> None:
        m = Machine(RANGES)
        assert m["gpu"] == Machine([RANGES[-1]])
        assert m["gpu", 1:2] == Machine([RANGES[-1].slice(slice(1, 2))])

        m = m.only("gpu")
        assert m[1] == Machine([RANGES[-1].slice(slice(1, 2))])
        assert m[1:] == Machine([RANGES[-1].slice(slice(1, None))])
        assert m[:2] == Machine([RANGES[-1].slice(slice(2))])

    def test_intersection(self) -> None:
        m1 = Machine(RANGES[:-1])
        m2 = Machine(RANGES[1:])
        assert m1 & m2 == Machine([RANGES[1]])

        m1 = Machine(RANGES[:-2])
        m2 = Machine(RANGES[2:])
        assert (m1 & m2).empty

    def test_empty(self) -> None:
        assert Machine([]).empty
        assert Machine([EMPTY_RANGE]).empty
        assert not Machine(RANGES).empty

    def test_pack(self) -> None:
        buf = BufferBuilder()
        Machine(RANGES).pack(buf)
        packed = buf.get_string()
        assert packed is not None
        assert buf.get_size() == 8 + len(RANGES) * 16
        values = struct.unpack("I" * 14, packed)
        assert values[0] == ProcessorKind.GPU
        assert values[1] == len(RANGES)
        values = values[2:]
        for i, r in enumerate(RANGES):
            v1, v2, v3, v4 = values[i * 4 : (i + 1) * 4]
            assert v1 == r.kind
            assert v2 == r.per_node_count
            assert v3 == r.lo
            assert v4 == r.hi

        buf = BufferBuilder()
        Machine([]).pack(buf)
        packed = buf.get_string()
        assert packed is not None
        assert buf.get_size() == 8
        values = struct.unpack("I" * 2, packed)
        assert values[0] == ProcessorKind.CPU
        assert values[1] == 0

    def test_idempotent_scopes(self) -> None:
        from legate.core import get_machine

        machine = get_machine()
        with machine:
            assert machine == get_machine()
            with machine:
                assert machine == get_machine()

    def test_machine_stack(self) -> None:
        from legate.core import get_machine
        from legate.core.runtime import runtime

        fake_machine = Machine(RANGES[-1:])
        runtime.push_machine(fake_machine)

        sub_machine1 = fake_machine[:-1]
        sub_machine2 = sub_machine1[1:]
        expected = sub_machine1[1:-1]

        orig_machine = get_machine()
        with sub_machine1:
            assert sub_machine1 == get_machine()
            with sub_machine2:
                assert expected == get_machine()
            assert sub_machine1 == get_machine()
        assert orig_machine == get_machine()

    def test_empty_scope(self) -> None:
        from legate.core import get_machine

        machine = get_machine()
        rng = machine.get_processor_range()
        empty_rng = ProcessorRange.create(rng.kind, rng.per_node_count, 1, 0)
        err_msg = "Empty machines cannot be used for resource scoping"
        with pytest.raises(ValueError, match=err_msg):
            with Machine([empty_rng]):
                pass
