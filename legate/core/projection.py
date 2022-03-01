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


from legion_cffi import ffi  # Make sure we only have one ffi instance


class ProjExpr(object):
    def __init__(self, dim=-1, weight=1, offset=0):
        self._dim = dim
        self._weight = weight
        self._offset = offset
        self._repr = None

    @property
    def dim(self):
        return self._dim

    @property
    def weight(self):
        return self._weight

    @property
    def offset(self):
        return self._offset

    def is_identity(self, dim):
        return self._dim == dim and self._weight == 1 and self._offset == 0

    def __repr__(self):
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

    def __hash__(self):
        return hash((type(self), self._dim, self._weight, self._offset))

    def __eq__(self, other):
        return (
            isinstance(other, ProjExpr)
            and self._dim == other._dim
            and self._weight == other._weight
            and self._offset == self._offset
        )

    def __mul__(self, other):
        if not isinstance(other, int):
            raise ValueError("RHS must be an integer")
        return ProjExpr(self._dim, self._weight * other, self._offset * other)

    def __add__(self, other):
        if not isinstance(other, int):
            raise ValueError("RHS must be an integer")
        return ProjExpr(self._dim, self._weight, self._offset + other)


def execute_functor_symbolically(ndim, proj_fn=None):
    point = tuple(ProjExpr(dim=dim) for dim in range(ndim))
    if proj_fn is not None:
        point = proj_fn(point)
        if not isinstance(point, tuple):
            raise ValueError("Projection function must return a tuple")

        point = tuple(
            ProjExpr(offset=v, weight=0) if isinstance(v, int) else v
            for v in point
        )
        if any(not isinstance(c, ProjExpr) for c in point):
            raise ValueError(
                "Each coordinate must be either a constant or "
                "one of the input coordinates"
            )

    return point


def is_identity_projection(src_ndims, dims):
    return src_ndims == len(dims) and all(
        isinstance(coord, ProjExpr) and coord.is_identity(dim)
        for dim, coord in enumerate(dims)
    )


def pack_symbolic_projection_repr(src_ndim, dims):
    tgt_ndim = len(dims)
    dims_c = ffi.new(f"int32_t[{tgt_ndim}]")
    weights_c = ffi.new(f"int32_t[{tgt_ndim}]")
    offsets_c = ffi.new(f"int32_t[{tgt_ndim}]")
    for dim, coord in enumerate(dims):
        dims_c[dim] = coord.dim
        weights_c[dim] = coord.weight
        offsets_c[dim] = coord.offset

    return (src_ndim, tgt_ndim, dims_c, weights_c, offsets_c)
