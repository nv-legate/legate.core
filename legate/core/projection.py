# Copyright 2021 NVIDIA Corporation
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
        return hash(repr(self))


def analyze_projection(ndim, proj_fn):
    point = tuple(CoordinateSym(i) for i in range(ndim))
    projected = proj_fn(point)
    if not isinstance(projected, tuple):
        raise ValueError("Projection function must return a tuple")
    elif any(not isinstance(c, (CoordinateSym, int)) for c in projected):
        raise ValueError(
            "Each coordinate must be either a constant or "
            "one of the input coordinates"
        )

    return projected


def pack_projection_spec(src_ndim, spec):
    tgt_ndim = len(spec)
    dims_c = ffi.new(f"int32_t[{tgt_ndim}]")
    offsets_c = ffi.new(f"int32_t[{tgt_ndim}]")
    for dim, coord in enumerate(spec):
        if isinstance(coord, CoordinateSym):
            dims_c[dim] = coord.dim
            offsets_c[dim] = 0
        else:
            dims_c[dim] = -1
            offsets_c[dim] = coord

    return (src_ndim, tgt_ndim, dims_c, offsets_c)
