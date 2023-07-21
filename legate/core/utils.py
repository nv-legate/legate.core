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

import traceback
from ctypes import CDLL, RTLD_GLOBAL
from types import TracebackType
from typing import (
    Any,
    Hashable,
    Iterable,
    Iterator,
    MutableSet,
    Optional,
    TypeVar,
)

T = TypeVar("T", bound="Hashable")


class OrderedSet(MutableSet[T]):
    """
    A set() variant whose iterator returns elements in insertion order.

    The implementation of this class piggybacks off of the corresponding
    iteration order guarantee for dict(), starting with Python 3.7. This is
    useful for guaranteeing symmetric execution of algorithms on different
    shards in a replicated context.
    """

    def __init__(self, copy_from: Optional[Iterable[T]] = None) -> None:
        self._dict: dict[T, None] = {}
        if copy_from is not None:
            for obj in copy_from:
                self.add(obj)

    def add(self, obj: T) -> None:
        self._dict[obj] = None

    def update(self, other: Iterable[T]) -> None:
        for obj in other:
            self.add(obj)

    def discard(self, obj: T) -> None:
        self._dict.pop(obj, None)

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, obj: object) -> bool:
        return obj in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    def remove_all(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return OrderedSet(obj for obj in self if obj not in other)


def cast_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else tuple(value)


def capture_traceback_repr(
    skip_core_frames: bool = True,
) -> Optional[str]:
    tb = None
    for frame, _ in traceback.walk_stack(None):
        if frame.f_globals["__name__"].startswith("legate.core"):
            continue
        tb = TracebackType(
            tb,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )
    return "".join(traceback.format_tb(tb)) if tb is not None else None


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))
