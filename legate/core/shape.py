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


from collections.abc import Iterable
from functools import reduce


def _cast_tuple(value, ndim):
    if isinstance(value, Shape):
        return value.extents
    elif isinstance(value, tuple):
        return value
    elif isinstance(value, int):
        return (value,) * ndim
    else:
        raise ValueError(f"Cannot cast {type(value).__name__} to tuple")


class Shape(object):
    def __init__(self, extents=None, ispace=None):
        if extents is not None:
            if not (
                isinstance(extents, Iterable) or isinstance(extents, Shape)
            ):
                self._extents = (extents,)
            else:
                self._extents = tuple(extents)
            self._ispace = None
        else:
            assert ispace is not None
            self._extents = None
            self._ispace = ispace

    @property
    def extents(self):
        if self._extents is None:
            assert self._ispace is not None
            bounds = self._ispace.get_bounds()
            lo = bounds.lo
            hi = bounds.hi
            assert all(lo[idx] == 0 for idx in range(lo.dim))
            self._extents = tuple(hi[idx] + 1 for idx in range(hi.dim))
        return self._extents

    def __str__(self):
        if self._extents is not None:
            return f"Shape({self._extents})"
        else:
            return f"Shape({self._ispace})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Shape(self.extents[idx])
        else:
            return self.extents[idx]

    def __len__(self):
        return len(self.extents)

    @property
    def fixed(self):
        return self._extents is not None

    @property
    def ispace(self):
        return self._ispace

    @property
    def ndim(self):
        return len(self.extents)

    def get_index_space(self, runtime):
        if self._ispace is None:
            bounds = self._extents
            assert bounds is not None
            return runtime.find_or_create_index_space(bounds)
        else:
            return self._ispace

    def volume(self):
        return reduce(lambda x, y: x * y, self.extents, 1)

    def sum(self):
        return reduce(lambda x, y: x + y, self.extents, 0)

    def __hash__(self):
        if self._ispace is not None:
            return hash((self.__class__, False, self._ispace))
        else:
            return hash((self.__class__, True, self._extents))

    def __eq__(self, other):
        if isinstance(other, Shape):
            if self._ispace is not None and other._ispace is not None:
                return self._ispace is other._ispace
            else:
                return self.extents == other.extents
        else:
            lh = _cast_tuple(self, self.ndim)
            rh = _cast_tuple(other, self.ndim)
            return lh == rh

    def __le__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh <= rh

    def __lt__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh < rh

    def __ge__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh >= rh

    def __gt__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh > rh

    def __add__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a + b for (a, b) in zip(lh, rh)))

    def __sub__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a - b for (a, b) in zip(lh, rh)))

    def __mul__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a * b for (a, b) in zip(lh, rh)))

    def __mod__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a % b for (a, b) in zip(lh, rh)))

    def __floordiv__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return Shape(tuple(a // b for (a, b) in zip(lh, rh)))

    def drop(self, dim):
        return Shape(self.extents[:dim] + self.extents[dim + 1 :])

    def update(self, dim, new_value):
        return self.replace(dim, (new_value,))

    def replace(self, dim, new_values):
        if not isinstance(new_values, tuple):
            new_values = tuple(new_values)
        return Shape(self.extents[:dim] + new_values + self.extents[dim + 1 :])

    def insert(self, dim, new_value):
        return Shape(self.extents[:dim] + (new_value,) + self.extents[dim:])

    def map(self, mapping):
        return Shape(tuple(self[mapping[dim]] for dim in range(self.ndim)))

    def strides(self):
        strides = ()
        stride = 1
        for size in reversed(self.extents):
            strides += (stride,)
            stride *= size
        return Shape(reversed(strides))
