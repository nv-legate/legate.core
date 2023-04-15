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

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

from . import legion, types as ty

if TYPE_CHECKING:
    from . import BufferBuilder
    from .runtime import Runtime


class EmptyMachineError(Exception):
    pass


# Make this consistent with TaskTarget in mapping.h
@unique
class ProcessorKind(IntEnum):
    GPU = 1
    OMP = 2
    CPU = 3


# Inclusive range of processor ids
@dataclass(frozen=True)
class ProcessorRange:
    # the kind is being used just in compatibility checks
    kind: ProcessorKind
    per_node_count: int
    lo: int
    hi: int

    @staticmethod
    def create(
        kind: ProcessorKind, per_node_count: int, lo: int, hi: int
    ) -> ProcessorRange:
        if hi < lo:
            lo = 1
            hi = 0
        return ProcessorRange(kind, per_node_count, lo, hi)

    @property
    def empty(self) -> bool:
        return self.hi < self.lo

    def __len__(self) -> int:
        return self.hi - self.lo + 1

    def __and__(self, other: ProcessorRange) -> ProcessorRange:
        if self.kind != other.kind:
            raise ValueError(
                "Intersection between different processor kinds: "
                f"{self.kind.name} and {other.kind.name}"
            )
        assert self.per_node_count == other.per_node_count
        return ProcessorRange.create(
            self.kind,
            self.per_node_count,
            max(self.lo, other.lo),
            min(self.hi, other.hi),
        )

    def slice(self, sl: slice) -> ProcessorRange:
        if sl.step is not None and sl.step != 1:
            raise ValueError("The slicing step must be 1")
        sz = len(self)
        new_lo = self.lo
        new_hi = self.hi
        if sl.start is not None:
            if sl.start >= 0:
                new_lo += sl.start
            else:
                new_lo += max(0, sl.start + sz)
        if sl.stop is not None:
            if sl.stop >= 0:
                new_hi = self.lo + sl.stop - 1
            else:
                new_hi = self.lo + max(0, sl.stop + sz)

        return ProcessorRange.create(
            self.kind, self.per_node_count, new_lo, new_hi
        )

    def get_node_range(self) -> tuple[int, int]:
        if self.empty:
            raise ValueError(
                "Illegal to get a node range of an empty processor range"
            )
        return (self.lo // self.per_node_count, self.hi // self.per_node_count)

    def __repr__(self) -> str:
        if self.hi < self.lo:
            return "<empty>"
        else:
            return f"[{self.lo}, {self.hi}] ({self.per_node_count} per node)"

    def pack(self, buf: BufferBuilder) -> None:
        buf.pack_32bit_uint(self.per_node_count)
        buf.pack_32bit_uint(self.lo)
        buf.pack_32bit_uint(self.hi)


ProcSlice = Tuple[ProcessorKind, slice]


class Machine:
    def __init__(self, proc_ranges: Sequence[ProcessorRange]) -> None:
        self._proc_ranges = dict((r.kind, r) for r in proc_ranges)
        self._non_empty_kinds = tuple(
            r.kind for r in proc_ranges if not r.empty
        )
        self._preferred_kind = min(
            self._non_empty_kinds, default=ProcessorKind.CPU
        )

    def __len__(self) -> int:
        return len(self._get_range(self._preferred_kind))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Machine):
            return False
        for kind in ProcessorKind:
            if self._get_range(kind) != other._get_range(kind):
                return False
        return True

    @property
    def preferred_kind(self) -> ProcessorKind:
        return self._preferred_kind

    @property
    def kinds(self) -> tuple[ProcessorKind, ...]:
        return self._non_empty_kinds

    def get_processor_range(
        self, kind: Optional[ProcessorKind] = None
    ) -> ProcessorRange:
        if kind is None:
            kind = self._preferred_kind
        return self._get_range(kind)

    def get_node_range(
        self, kind: Optional[ProcessorKind] = None
    ) -> tuple[int, int]:
        return self.get_processor_range(kind).get_node_range()

    def _get_range(self, kind: ProcessorKind) -> ProcessorRange:
        if kind not in self._proc_ranges:
            return ProcessorRange.create(kind, 1, 1, 0)
        return self._proc_ranges[kind]

    def only(self, *kinds: ProcessorKind) -> Machine:
        return Machine([self._get_range(kind) for kind in kinds])

    def count(self, kind: ProcessorKind) -> int:
        return len(self._get_range(kind))

    def filter_ranges(
        self, task_info: Any, variant_ids: dict[ProcessorKind, int]
    ) -> Machine:
        valid_kinds = tuple(
            kind
            for kind in self.kinds
            if task_info.has_variant(variant_ids[kind])
        )
        if valid_kinds == self.kinds:
            return self
        else:
            return self.only(*valid_kinds)

    def __getitem__(self, key: Union[str, slice, int, ProcSlice]) -> Machine:
        if isinstance(key, ProcessorKind):
            return self.only(key)
        elif isinstance(key, (slice, int)):
            if len(self._proc_ranges.keys()) > 1:
                raise ValueError(
                    "Ambiguous slicing: slicing is not allowed on a machine "
                    "with more than one processor kind"
                )
            k = key if isinstance(key, slice) else slice(key, key + 1)
            return Machine([self.get_processor_range().slice(k)])
        elif isinstance(key, tuple) and len(key) == 2:
            kind, sl = key
            if not isinstance(sl, slice):
                raise KeyError(f"Invalid slicing key: {key}")
            return Machine([self._get_range(kind).slice(sl)])
        else:
            raise KeyError(f"Invalid slicing key: {key}")

    @staticmethod
    def create_toplevel_machine(runtime: Runtime) -> Machine:
        num_nodes = int(
            runtime.core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_NUM_NODES,
                ty.int32,
            )
        )

        def create_range(kind: ProcessorKind) -> ProcessorRange:
            tunable_name = f"LEGATE_CORE_TUNABLE_TOTAL_{kind.name}S"
            tunable_id = getattr(legion, tunable_name)
            num_procs = int(
                runtime.core_context.get_tunable(tunable_id, ty.int32)
            )
            return ProcessorRange.create(
                kind, num_procs // num_nodes, 0, num_procs - 1
            )

        result = Machine([create_range(kind) for kind in ProcessorKind])
        if result.empty:
            raise RuntimeError(
                "No processors are available to run Legate tasks. Please "
                "create at least one processor of any kind. "
            )
        return result

    def __and__(self, other: Machine) -> Machine:
        if self is other:
            return self
        result: list[ProcessorRange] = []
        for kind, r in self._proc_ranges.items():
            if kind not in other._proc_ranges:
                continue
            result.append(r & other._proc_ranges[kind])
        return Machine(result)

    @property
    def empty(self) -> bool:
        return all(r.empty for r in self._proc_ranges.values())

    def __repr__(self) -> str:
        desc = ", ".join(
            f"{kind.name}: {prange}"
            for kind, prange in self._proc_ranges.items()
        )
        return f"Machine({desc})"

    def pack(self, buf: BufferBuilder) -> None:
        buf.pack_32bit_uint(self._preferred_kind)
        buf.pack_32bit_uint(len(self._proc_ranges))
        for kind, proc_range in self._proc_ranges.items():
            buf.pack_32bit_uint(kind)
            proc_range.pack(buf)

    def __enter__(self) -> None:
        from .runtime import runtime

        new_machine = runtime.machine & self
        if new_machine.empty:
            raise EmptyMachineError(
                "Empty machines cannot be used for resource scoping"
            )
        runtime.push_machine(new_machine)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        from .runtime import runtime

        runtime.pop_machine()
