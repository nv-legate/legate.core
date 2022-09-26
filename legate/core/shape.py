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

from functools import reduce
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Union, overload

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from . import IndexSpace
    from .runtime import Runtime

ExtentLike: TypeAlias = Union["Shape", int, Iterable[int]]


def _cast_tuple(value: ExtentLike, ndim: int) -> tuple[int, ...]:
    if isinstance(value, Shape):
        return value.extents
    elif isinstance(value, Iterable):
        return tuple(value)
    elif isinstance(value, int):
        return (value,) * ndim
    else:
        raise ValueError(f"Cannot cast {type(value).__name__} to tuple")


class Shape:
    _extents: Union[tuple[int, ...], None]
    _ispace: Union[IndexSpace, None]

    def __init__(
        self,
        extents: Optional[ExtentLike] = None,
        ispace: Optional[IndexSpace] = None,
    ) -> None:
        if extents is not None:
            self._extents = _cast_tuple(extents, 1)
            self._ispace = None
        else:
            assert ispace is not None
            self._extents = None
            self._ispace = ispace

    @property
    def extents(self) -> tuple[int, ...]:
        if self._extents is None:
            assert self._ispace is not None
            bounds = self._ispace.get_bounds()
            lo = bounds.lo
            hi = bounds.hi
            assert all(lo[idx] == 0 for idx in range(lo.dim))
            self._extents = tuple(hi[idx] + 1 for idx in range(hi.dim))
        return self._extents

    def __str__(self) -> str:
        if self._extents is not None:
            return f"Shape({self._extents})"
        else:
            return f"Shape({self._ispace})"

    def __repr__(self) -> str:
        return str(self)

    @overload
    def __getitem__(self, idx: int) -> int:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Shape:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Shape, int]:
        if isinstance(idx, slice):
            return Shape(self.extents[idx])
        else:
            return self.extents[idx]

    def __len__(self) -> int:
        return len(self.extents)

    def __iter__(self) -> Iterator[int]:
        return iter(self.extents)

    def __contains__(self, value: object) -> bool:
        return value in self.extents

    @property
    def fixed(self) -> bool:
        return self._extents is not None

    @property
    def ispace(self) -> Union[IndexSpace, None]:
        return self._ispace

    @property
    def ndim(self) -> int:
        if self._extents is None:
            assert self._ispace is not None
            return self._ispace.get_dim()
        else:
            return len(self._extents)

    def get_index_space(self, runtime: Runtime) -> IndexSpace:
        if self._ispace is None:
            bounds = self._extents
            assert bounds is not None
            # 0-D index spaces are invalid in Legion, so we have to promote
            # the bounds to 1-D
            if bounds == ():
                bounds = (1,)
            return runtime.find_or_create_index_space(bounds)
        else:
            return self._ispace

    def volume(self) -> int:
        return reduce(lambda x, y: x * y, self.extents, 1)

    def sum(self) -> int:
        return reduce(lambda x, y: x + y, self.extents, 0)

    def __hash__(self) -> int:
        if self._ispace is not None:
            return hash((self.__class__, False, self._ispace))
        else:
            return hash((self.__class__, True, self._extents))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Shape):
            if (
                self._ispace is not None
                and other._ispace is not None
                and self._ispace is other._ispace
            ):
                return True
            else:
                return self.extents == other.extents
        elif isinstance(other, (int, Iterable)):
            lh = _cast_tuple(self, self.ndim)
            rh = _cast_tuple(other, self.ndim)
            return lh == rh
        else:
            return False

    def __le__(self, other: ExtentLike) -> bool:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh <= rh

    def __lt__(self, other: ExtentLike) -> bool:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh < rh

    def __ge__(self, other: ExtentLike) -> bool:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh >= rh

    def __gt__(self, other: ExtentLike) -> bool:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh > rh

    def __add__(self, other: ExtentLike) -> Shape:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a + b for (a, b) in zip(lh, rh)))

    def __sub__(self, other: ExtentLike) -> Shape:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a - b for (a, b) in zip(lh, rh)))

    def __mul__(self, other: ExtentLike) -> Shape:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a * b for (a, b) in zip(lh, rh)))

    def __mod__(self, other: ExtentLike) -> Shape:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a % b for (a, b) in zip(lh, rh)))

    def __floordiv__(self, other: ExtentLike) -> Shape:
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a // b for (a, b) in zip(lh, rh)))

    def drop(self, dim: int) -> Shape:
        extents = self.extents
        return Shape(extents[:dim] + extents[dim + 1 :])

    def update(self, dim: int, new_value: int) -> Shape:
        return self.replace(dim, (new_value,))

    def replace(self, dim: int, new_values: Iterable[int]) -> Shape:
        if not isinstance(new_values, tuple):
            new_values = tuple(new_values)
        extents = self.extents
        return Shape(extents[:dim] + new_values + extents[dim + 1 :])

    def insert(self, dim: int, new_value: int) -> Shape:
        extents = self.extents
        return Shape(extents[:dim] + (new_value,) + extents[dim:])

    def map(self, mapping: tuple[int, ...]) -> Shape:
        return Shape(tuple(self[mapping[dim]] for dim in range(self.ndim)))

    def strides(self) -> Shape:
        strides: tuple[int, ...] = ()
        stride = 1
        for size in reversed(self.extents):
            strides += (stride,)
            stride *= size
        return Shape(reversed(strides))
