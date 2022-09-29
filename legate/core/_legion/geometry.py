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

from typing import Any, Collection, Iterator, Optional, Union

from .. import ffi, legion
from .env import LEGATE_MAX_DIM


class Point:
    point: Any

    def __init__(
        self, p: Optional[Union[Point, Any]] = None, dim: Optional[int] = None
    ) -> None:
        """
        The Point class wraps a `legion_domain_point_t` in the Legion C API.
        """
        if dim is None:
            self.point = legion.legion_domain_point_origin(0)
        else:
            self.point = legion.legion_domain_point_origin(dim)
        if p is not None:
            self.set_point(p)

    @property
    def dim(self) -> int:
        return self.point.dim

    def __getitem__(self, key: int) -> Any:
        if key >= self.dim:
            raise KeyError("key cannot exceed dimensionality")
        return self.point.point_data[key]

    def __setitem__(self, key: int, value: Any) -> None:
        if key >= self.dim:
            raise KeyError("key cannot exceed dimensionality")
        self.point.point_data[key] = value

    def __hash__(self) -> int:
        value = hash(self.dim)
        for idx in range(self.dim):
            value = value ^ hash(self[idx])
        return value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        if self.dim != len(other):
            return False
        for idx in range(self.dim):
            if self[idx] != other[idx]:
                return False
        return True

    def __len__(self) -> int:
        return self.dim

    def __iter__(self) -> Iterator[int]:
        for idx in range(self.dim):
            yield self[idx]

    def __repr__(self) -> str:
        p_strs = [str(self[i]) for i in range(self.dim)]
        return "Point(p=[" + ",".join(p_strs) + "])"

    def __str__(self) -> str:
        p_strs = [str(self[i]) for i in range(self.dim)]
        return "<" + ",".join(p_strs) + ">"

    def set_point(self, p: Any) -> None:
        try:
            if ffi.typeof(p).cname == "legion_domain_point_t":
                ffi.addressof(self.point)[0] = p
                return
        except TypeError:
            pass
        try:
            if len(p) > LEGATE_MAX_DIM:
                raise ValueError(
                    "Point cannot exceed "
                    + str(LEGATE_MAX_DIM)
                    + " dimensions set from LEGATE_MAX_DIM"
                )
            self.point.dim = len(p)
            for i, x in enumerate(p):
                self.point.point_data[i] = x
        except TypeError:
            self.point.dim = 1
            self.point.point_data[0] = p

    def raw(self) -> Any:
        return self.point


class Rect:
    def __init__(
        self,
        hi: Optional[Collection[int]] = None,
        lo: Optional[Collection[int]] = None,
        exclusive: bool = True,
        dim: Optional[int] = None,
    ) -> None:
        """
        The Rect class represents an N-D rectangle of dense points. It wraps a
        dense `legion_domain_t` (this is a special case for Domains; in the
        general case a Domain can also contain a sparsity map).
        """
        self._lo = Point(dim=dim)
        self._hi = Point(dim=dim)
        if dim is None:
            self.rect = legion.legion_domain_empty(0)
        else:
            self.rect = legion.legion_domain_empty(dim)
        if hi is not None:
            self.set_bounds(lo=lo, hi=hi, exclusive=exclusive)
        elif lo is not None:
            raise ValueError("'lo' cannot be set without 'hi'")

    @property
    def lo(self) -> Point:
        return self._lo

    @property
    def hi(self) -> Point:
        return self._hi

    @property
    def dim(self) -> int:
        assert self._lo.dim == self._hi.dim
        return self._lo.dim

    def get_volume(self) -> int:
        volume = 1
        for i in range(self.dim):
            volume *= self.hi[i] - self.lo[i] + 1
        return volume

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rect):
            return NotImplemented
        if self.lo != other.lo:
            return False
        if self.hi != other.hi:
            return False
        return True

    def __hash__(self) -> int:
        result = hash(self.dim)
        for idx in range(self.dim):
            result = result ^ hash(self.lo[idx])
            result = result ^ hash(self.hi[idx])
        return result

    def __iter__(self) -> Iterator[Point]:
        p = Point(self._lo)
        dim = self.dim
        yield Point(p)
        while True:
            for idx in range(dim - 1, -2, -1):
                if idx < 0:
                    return
                if p[idx] < self._hi[idx]:
                    p[idx] += 1
                    yield Point(p)
                    break
                p[idx] = self._lo[idx]

    def __repr__(self) -> str:
        return f"Rect(lo={repr(self._lo)},hi={repr(self._hi)},exclusive=False)"

    def __str__(self) -> str:
        return str(self._lo) + ".." + str(self._hi)

    def set_bounds(
        self,
        lo: Optional[Collection[int]],
        hi: Collection[int],
        exclusive: bool = True,
    ) -> None:
        if len(hi) > LEGATE_MAX_DIM:
            raise ValueError(
                "Point cannot exceed "
                + str(LEGATE_MAX_DIM)
                + " dimensions set from LEGATE_MAX_DIM"
            )
        if exclusive:
            self._hi.set_point([x - 1 for x in hi])
        else:
            self._hi.set_point(hi)
        if lo is not None:
            if len(lo) != len(hi):
                raise ValueError("Length of 'lo' must equal length of 'hi'")
            self._lo.set_point(lo)
        else:
            self._lo.set_point((0,) * len(hi))

    def raw(self) -> Any:
        dim = self._hi.dim
        self.rect.dim = dim
        for i in range(dim):
            self.rect.rect_data[i] = self._lo[i]
            self.rect.rect_data[dim + i] = self._hi[i]
        return self.rect

    def to_domain(self) -> Domain:
        return Domain(self.raw())


class Domain:
    def __init__(self, domain: Any) -> None:
        """
        The Domain class wraps a `legion_domain_t` in the Legion C API. A
        Domain is the value stored by an IndexSpace. It consists of an N-D
        rectangle describing an upper bound on the points contained in the
        IndexSpace as well as optional sparsity map describing the actual
        non-dense set of points. If there is no sparsity map then the domain
        is purely the set of dense points represented by the rectangle bounds.

        Note that Domain objects do not copy the contents of the provided
        `legion_domain_t` handle, nor do they take ownership of if. It is up
        to the calling code to ensure that the memory backing the original
        handle will not be collected while this object is in use.
        """

        self.domain = domain
        self.rect = Rect(dim=domain.dim)
        for i in range(domain.dim):
            self.rect.lo[i] = self.domain.rect_data[i]
            self.rect.hi[i] = self.domain.rect_data[domain.dim + i]
        self.dense = legion.legion_domain_is_dense(domain)

    @property
    def dim(self) -> int:
        return self.rect.dim

    def get_volume(self) -> float:
        return legion.legion_domain_get_volume(self.domain)

    def get_rects(self) -> list[Rect]:
        # NOTE: For debugging only!
        create = getattr(
            legion,
            f"legion_rect_in_domain_iterator_create_{self.dim}d",
        )
        destroy = getattr(
            legion,
            f"legion_rect_in_domain_iterator_destroy_{self.dim}d",
        )
        valid = getattr(
            legion,
            f"legion_rect_in_domain_iterator_valid_{self.dim}d",
        )
        step = getattr(
            legion,
            f"legion_rect_in_domain_iterator_step_{self.dim}d",
        )
        get_rect = getattr(
            legion,
            f"legion_rect_in_domain_iterator_get_rect_{self.dim}d",
        )
        rects = []
        iterator = create(self.domain)
        while valid(iterator):
            nd_rect = get_rect(iterator)
            lo = [nd_rect.lo.x[i] for i in range(self.dim)]
            hi = [nd_rect.hi.x[i] for i in range(self.dim)]
            rects.append(Rect(hi=hi, lo=lo, exclusive=False, dim=self.dim))
            step(iterator)
        destroy(iterator)
        return rects
