# Copyright 2023 NVIDIA Corporation
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
    EmptyMachineError,
    Machine,
    ProcessorKind,
    ProcessorRange,
    ProcessorSlice,
)


class TestProcessorKind:
    def test_names(self) -> None:
        assert set(k.name for k in ProcessorKind) == {"GPU", "OMP", "CPU"}

    def test_values(self) -> None:
        assert list(ProcessorKind) == [1, 2, 3]


class TestProcessorRange:
    def test_create_nonempty(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.CPU, low=1, high=3, per_node_count=1
        )
        assert not r.empty
        assert r.kind == ProcessorKind.CPU
        assert r.per_node_count == 1 and r.low == 1 and r.high == 3
        assert len(r) == 2

        assert r.get_node_range() == (1, 3)

    def test_create_empty(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.GPU, low=1, high=0, per_node_count=1
        )
        assert r.empty
        assert r.kind == ProcessorKind.GPU
        assert r.per_node_count == 1 and r.low == 0 and r.high == 0
        assert len(r) == 0

        r = ProcessorRange.create(
            ProcessorKind.GPU, low=2, high=1, per_node_count=1
        )
        assert r.empty
        assert r.low == 0 and r.high == 0
        assert len(r) == 0

        r = ProcessorRange.create_empty_range(ProcessorKind.GPU)
        assert r.empty
        assert r.low == 0 and r.high == 0
        assert len(r) == 0

        err_msg = "Illegal to get a node range of an empty processor range"
        with pytest.raises(ValueError, match=err_msg):
            r.get_node_range()

    def test_intersection_nonempty(self) -> None:
        r1 = ProcessorRange.create(
            ProcessorKind.GPU, low=0, high=3, per_node_count=1
        )
        r2 = ProcessorRange.create(
            ProcessorKind.GPU, low=2, high=4, per_node_count=1
        )
        r3 = r1 & r2
        assert r3 == ProcessorRange.create(
            ProcessorKind.GPU, low=2, high=3, per_node_count=1
        )

    def test_intersection_empty(self) -> None:
        r1 = ProcessorRange.create(
            ProcessorKind.GPU, low=0, high=2, per_node_count=1
        )
        r2 = ProcessorRange.create(
            ProcessorKind.GPU, low=3, high=5, per_node_count=1
        )
        r3 = r1 & r2
        assert r3 == ProcessorRange.create(
            ProcessorKind.GPU, low=1, high=0, per_node_count=1
        )
        assert len(r3) == 0

    def test_intersection_invalid(self) -> None:
        r1 = ProcessorRange.create(
            ProcessorKind.CPU, low=1, high=0, per_node_count=1
        )
        r2 = ProcessorRange.create(
            ProcessorKind.GPU, low=1, high=0, per_node_count=1
        )
        err_msg = (
            "Intersection between different processor kinds: " "CPU and GPU"
        )
        with pytest.raises(ValueError, match=err_msg):
            r1 & r2

    def test_empty_slice_empty_range(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.GPU, low=3, high=1, per_node_count=1
        )
        assert len(r[0:0]) == 0
        assert len(r.slice(slice(0, 0))) == 0
        assert len(r[:0]) == 0
        assert len(r.slice(slice(None, 0))) == 0
        assert len(r[4:]) == 0
        assert len(r.slice(slice(4, None))) == 0

    def test_empty_slice_nonempty_range(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.GPU, low=2, high=5, per_node_count=1
        )
        assert len(r[0:0]) == 0
        assert len(r.slice(slice(0, 0))) == 0
        assert len(r[:0]) == 0
        assert len(r.slice(slice(None, 0))) == 0
        assert len(r[5:]) == 0
        assert len(r.slice(slice(5, None))) == 0

    def test_nonempty_slice_empty_range(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.GPU, low=3, high=1, per_node_count=1
        )
        assert len(r[:]) == 0
        assert len(r.slice(slice(None))) == 0
        for i in range(len(r)):
            assert len(r[:i]) == 0
            assert len(r.slice(slice(i))) == 0
            assert len(r[i:]) == 0
            assert len(r.slice(slice(i, None))) == 0

    def test_nonempty_slice_nonempty_range(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.GPU, low=3, high=5, per_node_count=1
        )
        assert len(r[:]) == len(r)
        assert len(r.slice(slice(None))) == len(r)
        for i in range(len(r)):
            assert len(r[:i]) == i
            assert len(r.slice(slice(i))) == i
            assert len(r[i:]) == len(r) - i
            assert len(r.slice(slice(i, None))) == len(r) - i

    def test_invalid_slice(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.GPU, low=2, high=4, per_node_count=1
        )
        with pytest.raises(ValueError, match="The slicing step must be 1"):
            r.slice(slice(None, None, 2))

    def test_pack(self) -> None:
        r = ProcessorRange.create(
            ProcessorKind.CPU, low=3, high=4, per_node_count=5
        )
        buf = BufferBuilder()
        r.pack(buf)
        packed = buf.get_string()
        assert packed is not None
        assert buf.get_size() == 12
        v1, v2, v3 = struct.unpack("III", packed)
        assert v1 == 3
        assert v2 == 4
        assert v3 == 5


CPU_RANGE = ProcessorRange.create(
    ProcessorKind.CPU, low=1, high=3, per_node_count=4
)
OMP_RANGE = ProcessorRange.create(
    ProcessorKind.OMP, low=0, high=3, per_node_count=2
)
GPU_RANGE = ProcessorRange.create(
    ProcessorKind.GPU, low=3, high=6, per_node_count=3
)
EMPTY_RANGE = ProcessorRange.create_empty_range(ProcessorKind.GPU)
RANGES = [CPU_RANGE, OMP_RANGE, GPU_RANGE]


class TestMachine:
    def test_empty_machine(self) -> None:
        m = Machine([])
        assert m.preferred_kind == ProcessorKind.CPU
        assert len(m) == 0
        assert len(m.get_processor_range(ProcessorKind.CPU)) == 0
        err_msg = "Illegal to get a node range of an empty processor range"
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range()
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range(ProcessorKind.GPU)
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range(ProcessorKind.OMP)
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range(ProcessorKind.CPU)

    def test_eq(self) -> None:
        m1 = Machine([CPU_RANGE, OMP_RANGE])
        m2 = Machine([CPU_RANGE, OMP_RANGE])
        assert m1 == m2

        m3 = Machine([])
        assert m1 != m3

        m4 = Machine([CPU_RANGE, OMP_RANGE, EMPTY_RANGE])
        assert m1 == m4

    @pytest.mark.parametrize("n", range(1, len(RANGES) + 1))
    def test_preferred_kind(self, n: int) -> None:
        m = Machine(RANGES[:n])
        assert m.preferred_kind == RANGES[n - 1].kind
        assert len(m) == len(RANGES[n - 1])

    @pytest.mark.parametrize("n", range(1, len(RANGES) + 1))
    def test_kinds(self, n: int) -> None:
        m = Machine(RANGES[:n])
        assert m.kinds == tuple(r.kind for r in RANGES[:n])

    def test_get_processor_range(self) -> None:
        m = Machine([CPU_RANGE, OMP_RANGE])
        assert m.get_processor_range(ProcessorKind.CPU) == CPU_RANGE
        assert m.get_processor_range(ProcessorKind.OMP) == OMP_RANGE
        assert m.get_processor_range() == OMP_RANGE
        assert len(m.get_processor_range(ProcessorKind.GPU)) == 0

    def test_get_node_range(self) -> None:
        m = Machine(RANGES)
        assert m.get_node_range(ProcessorKind.CPU) == (0, 0)
        assert m.get_node_range(ProcessorKind.OMP) == (0, 1)
        assert m.get_node_range(ProcessorKind.GPU) == (1, 2)
        assert m.get_node_range() == m.get_node_range(ProcessorKind.GPU)

    def test_only(self) -> None:
        m = Machine(RANGES)
        gpu = ProcessorKind.GPU
        cpu = ProcessorKind.CPU
        omp = ProcessorKind.OMP
        assert len(m.only(gpu)) == len(GPU_RANGE)
        assert m.only(gpu).only(gpu) == m.only(gpu)
        assert len(m.only(gpu).get_processor_range(cpu)) == 0
        assert m.only(gpu).get_processor_range(gpu) == GPU_RANGE
        assert len(m.only(gpu, cpu)) == len(GPU_RANGE)
        assert len(m.only(gpu, cpu).only(gpu)) == len(GPU_RANGE)
        assert len(m.only(gpu, cpu).only(cpu)) == len(CPU_RANGE)
        assert len(m.only(gpu, cpu).only(omp)) == 0

    def test_count(self) -> None:
        m = Machine(RANGES)
        assert m.count(ProcessorKind.CPU) == len(CPU_RANGE)
        assert m.count(ProcessorKind.OMP) == len(OMP_RANGE)
        assert m.count(ProcessorKind.GPU) == len(GPU_RANGE)

    def test_get_item(self) -> None:
        m = Machine(RANGES)
        assert m[ProcessorKind.GPU] == Machine([GPU_RANGE])
        assert m[ProcessorSlice(ProcessorKind.GPU, slice(1, 2))] == Machine(
            [GPU_RANGE[1:2]]
        )

        m = m.only(ProcessorKind.GPU)
        assert m[4] == Machine([GPU_RANGE[4]])
        assert m[4:] == Machine([GPU_RANGE[4:]])
        assert m[:5] == Machine([GPU_RANGE[:5]])

    def test_intersection(self) -> None:
        m1 = Machine([CPU_RANGE, OMP_RANGE])
        m2 = Machine([OMP_RANGE, GPU_RANGE])
        assert m1 & m2 == Machine([OMP_RANGE])

        m1 = Machine([CPU_RANGE])
        m2 = Machine([OMP_RANGE])
        assert (m1 & m2).empty

    def test_empty(self) -> None:
        assert Machine([]).empty
        assert Machine([EMPTY_RANGE]).empty
        assert not Machine(RANGES).empty
        assert Machine([]).kinds == tuple()
        assert Machine([EMPTY_RANGE]).kinds == tuple()

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
            assert v2 == r.low
            assert v3 == r.high
            assert v4 == r.per_node_count

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

        fake_machine = Machine([GPU_RANGE])
        runtime.push_machine(fake_machine)

        sub_machine1 = fake_machine[:-1]
        sub_machine2 = sub_machine1[1:]
        expected = fake_machine[1:-1]

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
        empty_rng = ProcessorRange.create(
            rng.kind, low=1, high=0, per_node_count=rng.per_node_count
        )
        err_msg = "Empty machines cannot be used for resource scoping"
        with pytest.raises(EmptyMachineError, match=err_msg):
            with Machine([empty_rng]):
                pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
