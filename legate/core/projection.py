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


class CoordinateSym(object):
    def __init__(self, index):
        self._index = index

    @property
    def dim(self):
        return self._index

    def __repr__(self):
        return f"COORD{self._index}"

    def __hash__(self):
        return hash((type(self), self._index))

    def __eq__(self, other):
        return isinstance(other, CoordinateSym) and self._index == other._index


def execute_functor_symbolically(ndim, proj_fn=None):
    point = tuple(CoordinateSym(i) for i in range(ndim))
    if proj_fn is not None:
        point = proj_fn(point)
        if not isinstance(point, tuple):
            raise ValueError("Projection function must return a tuple")
        elif any(not isinstance(c, (CoordinateSym, int)) for c in point):
            raise ValueError(
                "Each coordinate must be either a constant or "
                "one of the input coordinates"
            )

    return point


def is_identity_projection(src_ndims, dims):
    return src_ndims == len(dims) and all(
        isinstance(coord, CoordinateSym) and dim == coord.dim
        for dim, coord in enumerate(dims)
    )


def pack_symbolic_projection_repr(src_ndim, dims):
    tgt_ndim = len(dims)
    dims_c = ffi.new(f"int32_t[{tgt_ndim}]")
    offsets_c = ffi.new(f"int32_t[{tgt_ndim}]")
    for dim, coord in enumerate(dims):
        if isinstance(coord, CoordinateSym):
            dims_c[dim] = coord.dim
            offsets_c[dim] = 0
        else:
            dims_c[dim] = -1
            offsets_c[dim] = coord

    return (src_ndim, tgt_ndim, dims_c, offsets_c)
