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

from typing import Any, Callable, Optional, Tuple, Union

from . import ffi  # Make sure we only have one ffi instance


class ProjExpr:
    def __init__(
        self, dim: int = -1, weight: int = 1, offset: int = 0
    ) -> None:
        self._dim = dim
        self._weight = weight
        self._offset = offset
        self._repr: Union[str, None] = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def weight(self) -> int:
        return self._weight

    @property
    def offset(self) -> int:
        return self._offset

    def is_identity(self, dim: int) -> bool:
        return self._dim == dim and self._weight == 1 and self._offset == 0

    def __repr__(self) -> str:
        if self._repr is None:
            s = ""
            if self._weight != 0:
                if self._weight != 1:
                    s += f"{self._weight} * "
                s += f"COORD{self._dim}"
            if self._offset != 0:
                if self._offset > 0:
                    s += f" + {self._offset}"
                else:
                    s += f" - {abs(self._offset)}"
            self._repr = s
        return self._repr

    def __hash__(self) -> int:
        return hash((type(self), self._dim, self._weight, self._offset))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjExpr):
            return NotImplemented
        return (
            self._dim == other._dim
            and self._weight == other._weight
            and self._offset == self._offset
        )

    def __mul__(self, other: int) -> ProjExpr:
        if not isinstance(other, int):
            raise ValueError("RHS must be an integer")
        return ProjExpr(self._dim, self._weight * other, self._offset * other)

    def __add__(self, other: int) -> ProjExpr:
        if not isinstance(other, int):
            raise ValueError("RHS must be an integer")
        return ProjExpr(self._dim, self._weight, self._offset + other)


# todo: (bev) use tuple[...] when feasible
SymbolicPoint = Tuple[ProjExpr, ...]

ProjOut = Tuple[Union[int, ProjExpr], ...]

ProjFn = Callable[[SymbolicPoint], ProjOut]


def execute_functor_symbolically(
    ndim: int, proj_fn: Optional[ProjFn] = None
) -> SymbolicPoint:
    point: SymbolicPoint = tuple(ProjExpr(dim=dim) for dim in range(ndim))
    if proj_fn is not None:
        result = proj_fn(point)
        if not isinstance(point, tuple):
            raise ValueError("Projection function must return a tuple")

        point = tuple(
            ProjExpr(offset=v, weight=0) if isinstance(v, int) else v
            for v in result
        )
        if any(not isinstance(c, ProjExpr) for c in point):
            raise ValueError(
                "Each coordinate must be either a constant or "
                "one of the input coordinates"
            )

    return point


def is_identity_projection(src_ndims: int, dims: SymbolicPoint) -> bool:
    return src_ndims == len(dims) and all(
        isinstance(coord, ProjExpr) and coord.is_identity(dim)
        for dim, coord in enumerate(dims)
    )


def pack_symbolic_projection_repr(
    src_ndim: int, dims: tuple[ProjExpr, ...]
) -> tuple[int, int, Any, Any, Any]:
    tgt_ndim = len(dims)
    dims_c = ffi.new(f"int32_t[{tgt_ndim}]")
    weights_c = ffi.new(f"int32_t[{tgt_ndim}]")
    offsets_c = ffi.new(f"int32_t[{tgt_ndim}]")
    for dim, coord in enumerate(dims):
        dims_c[dim] = coord.dim
        weights_c[dim] = coord.weight
        offsets_c[dim] = coord.offset

    return (src_ndim, tgt_ndim, dims_c, weights_c, offsets_c)
