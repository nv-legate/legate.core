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

from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, Tuple, Union

from . import legion, types as ty

if TYPE_CHECKING:
    from . import BufferBuilder
    from .runtime import Runtime


# Make this consistent with TaskTarget in mapping.h
@unique
class ProcessorKind(IntEnum):
    GPU = 1
    OMP = 2
    CPU = 3


def sanitize_kind(kind_str: Union[str, ProcessorKind]) -> ProcessorKind:
    return (
        getattr(ProcessorKind, kind_str.upper())
        if isinstance(kind_str, str)
        else kind_str
    )


PRECEDENCE = (
    ProcessorKind.GPU,
    ProcessorKind.OMP,
    ProcessorKind.CPU,
)


# Inclusive range of processor ids
class ProcessorRange:
    def __init__(self, lo: int, hi: int) -> None:
        if hi < lo:
            lo = 1
            hi = 0
        self.lo = lo
        self.hi = hi

    def __len__(self) -> int:
        return self.hi - self.lo + 1

    def __and__(self, other: ProcessorRange) -> ProcessorRange:
        return ProcessorRange(max(self.lo, other.lo), min(self.hi, other.hi))

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

        return ProcessorRange(new_lo, new_hi)

    def __str__(self) -> str:
        if self.hi < self.lo:
            return "<empty>"
        else:
            return f"[{self.lo}, {self.hi}]"

    def __repr__(self) -> str:
        return str(self)

    def pack(self, buf: BufferBuilder) -> None:
        buf.pack_32bit_uint(self.lo)
        buf.pack_32bit_uint(self.hi)


EMPTY_RANGE = ProcessorRange(1, 0)


ProcKindLike = Union[str, ProcessorKind]
ProcSlice = Tuple[ProcKindLike, slice]


class Machine:
    def __init__(
        self,
        runtime: Runtime,
        proc_ranges: dict[ProcessorKind, ProcessorRange],
    ) -> None:
        self._runtime = runtime
        self._proc_ranges = proc_ranges
        self._preferred_kind = ProcessorKind.CPU
        for kind in PRECEDENCE:
            if kind in proc_ranges and len(proc_ranges[kind]) > 0:
                self._preferred_kind = kind
                break
        self._num_procs = len(self._proc_ranges[self._preferred_kind])

    @property
    def num_procs(self) -> int:
        return self._num_procs

    @property
    def preferred_kind(self) -> ProcessorKind:
        return self._preferred_kind

    def _get_range(self, kind: ProcessorKind) -> ProcessorRange:
        return self._proc_ranges.get(kind, EMPTY_RANGE)

    def only(self, kind: ProcKindLike) -> Machine:
        sanitized = sanitize_kind(kind)
        return Machine(self._runtime, {sanitized: self._get_range(sanitized)})

    def exclude(self, kind: ProcKindLike) -> Machine:
        sanitized = sanitize_kind(kind)
        ranges = self._proc_ranges.copy()
        if sanitized in ranges:
            del ranges[sanitized]
        return Machine(self._runtime, ranges)

    def count(self, kind: ProcKindLike) -> int:
        sanitized = sanitize_kind(kind)
        return len(self._get_range(sanitized))

    def __getitem__(self, key: Union[str, slice, int, ProcSlice]) -> Machine:
        if isinstance(key, str):
            return self.only(key)
        elif isinstance(key, (slice, int)):
            if len(self._proc_ranges.keys()) > 1:
                raise ValueError(
                    "Ambiguous slicing: slicing is not allowed on a machine "
                    "with more than one processor kind"
                )
            k = key if isinstance(key, slice) else slice(key, key + 1)
            kind = self._preferred_kind
            return Machine(
                self._runtime, {kind: self._get_range(kind).slice(k)}
            )
        elif isinstance(key, tuple) and len(key) == 2:
            kind = sanitize_kind(key[0])
            new_ranges = self._proc_ranges.copy()
            new_ranges[kind] = self._get_range(kind).slice(key[1])
            return Machine(self._runtime, new_ranges)
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

        def create_range(tunable: int) -> ProcessorRange:
            num_procs = int(
                runtime.core_context.get_tunable(tunable, ty.int32)
            )
            return ProcessorRange(0, num_nodes * num_procs - 1)

        ranges = {
            ProcessorKind.GPU: create_range(
                legion.LEGATE_CORE_TUNABLE_TOTAL_GPUS
            ),
            ProcessorKind.OMP: create_range(
                legion.LEGATE_CORE_TUNABLE_TOTAL_OMPS
            ),
            ProcessorKind.CPU: create_range(
                legion.LEGATE_CORE_TUNABLE_TOTAL_CPUS
            ),
        }

        result = Machine(runtime, ranges)
        if result.empty:
            raise RuntimeError(
                "No processors are available to run Legate tasks. Please "
                "enable at least one processor of any kind. "
            )
        return result

    def __and__(self, other: Machine) -> Machine:
        result: dict[ProcessorKind, ProcessorRange] = {}
        for kind, prange in self._proc_ranges.items():
            if kind not in other._proc_ranges:
                continue
            result[kind] = prange & other._proc_ranges[kind]
        return Machine(self._runtime, result)

    @property
    def empty(self) -> bool:
        return all(len(prange) == 0 for prange in self._proc_ranges.values())

    def __str__(self) -> str:
        desc = ", ".join(
            f"{kind.name}: {prange}"
            for kind, prange in self._proc_ranges.items()
        )
        return f"Machine({desc})"

    def __repr__(self) -> str:
        return str(self)

    def pack(self, buf: BufferBuilder) -> None:
        buf.pack_32bit_uint(self._preferred_kind)
        buf.pack_32bit_uint(len(self._proc_ranges))
        for kind, proc_range in self._proc_ranges.items():
            buf.pack_32bit_uint(kind)
            proc_range.pack(buf)

    def __enter__(self) -> None:
        new_machine = self._runtime.machine & self
        if new_machine.empty:
            raise ValueError(
                "Empty machines cannot be used for resource scoping"
            )
        self._runtime.push_machine(new_machine)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        self._runtime.pop_machine()
