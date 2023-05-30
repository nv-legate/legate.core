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
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

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


@dataclass(frozen=True)
class ProcessorSlice:
    kind: ProcessorKind
    slice: slice


PROC_RANGE_KEY = Union[slice, int]
MACHINE_KEY = Union[ProcessorKind, slice, int, ProcessorSlice]


# Inclusive range of processor ids
@dataclass(frozen=True)
class ProcessorRange:
    # the kind is being used just in compatibility checks
    kind: ProcessorKind
    low: int
    high: int
    per_node_count: int

    @staticmethod
    def create(
        kind: ProcessorKind,
        *,
        low: int,
        high: int,
        per_node_count: int,
    ) -> ProcessorRange:
        if high <= low:
            low = 0
            high = 0
        return ProcessorRange(kind, low, high, per_node_count)

    @staticmethod
    def create_empty_range(kind: ProcessorKind) -> ProcessorRange:
        return ProcessorRange(kind, 0, 0, 1)

    @property
    def empty(self) -> bool:
        return self.high <= self.low

    def __len__(self) -> int:
        return self.high - self.low

    def __and__(self, other: ProcessorRange) -> ProcessorRange:
        if self.kind != other.kind:
            raise ValueError(
                "Intersection between different processor kinds: "
                f"{self.kind.name} and {other.kind.name}"
            )
        assert self.per_node_count == other.per_node_count
        return ProcessorRange.create(
            self.kind,
            low=max(self.low, other.low),
            high=min(self.high, other.high),
            per_node_count=self.per_node_count,
        )

    def slice(self, sl: slice) -> ProcessorRange:
        if sl.step is not None and sl.step != 1:
            raise ValueError("The slicing step must be 1 or None")
        sz = len(self)
        new_low = self.low
        new_high = self.high
        if sl.start is not None:
            if sl.start >= 0:
                new_low += sl.start
            else:
                new_low += max(0, sl.start + sz)
        if sl.stop is not None:
            if sl.stop >= 0:
                new_high = self.low + sl.stop
            else:
                new_high = self.low + max(0, sl.stop + sz)

        return ProcessorRange.create(
            self.kind,
            low=new_low,
            high=new_high,
            per_node_count=self.per_node_count,
        )

    def __getitem__(self, key: PROC_RANGE_KEY) -> ProcessorRange:
        if isinstance(key, int):
            return self.slice(slice(key, key + 1))
        elif isinstance(key, slice):
            return self.slice(key)

        raise KeyError(f"Invalid slicing key: {key}")

    def get_node_range(self) -> tuple[int, int]:
        if self.empty:
            raise ValueError(
                "Illegal to get a node range of an empty processor range"
            )
        return (
            self.low // self.per_node_count,
            (self.high + self.per_node_count - 1) // self.per_node_count,
        )

    def __repr__(self) -> str:
        if self.empty:
            return "<empty>"
        return f"[{self.low}, {self.high}] ({self.per_node_count} per node)"

    def pack(self, buf: BufferBuilder) -> None:
        buf.pack_32bit_uint(self.low)
        buf.pack_32bit_uint(self.high)
        buf.pack_32bit_uint(self.per_node_count)


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
            return ProcessorRange.create_empty_range(kind)
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

    def __getitem__(self, key: MACHINE_KEY) -> Machine:
        if isinstance(key, ProcessorKind):
            return self.only(key)
        elif isinstance(key, (slice, int)):
            if len(self._non_empty_kinds) > 1:
                raise ValueError(
                    "Ambiguous slicing: slicing is not allowed on a machine "
                    "with more than one processor kind"
                )
            return Machine([self.get_processor_range()[key]])
        elif isinstance(key, ProcessorSlice):
            return Machine(
                [self.get_processor_range(key.kind).slice(key.slice)]
            )

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
                kind,
                low=0,
                high=num_procs,
                per_node_count=num_procs // num_nodes,
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
        result = [
            r & other._proc_ranges[kind]
            for kind, r in self._proc_ranges.items()
            if kind in other._proc_ranges
        ]
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
        buf.pack_32bit_int(self._preferred_kind)
        buf.pack_32bit_uint(len(self._proc_ranges))
        for kind, proc_range in self._proc_ranges.items():
            buf.pack_32bit_int(kind)
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
