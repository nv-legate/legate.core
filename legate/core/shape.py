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


from collections.abc import Iterable
from functools import reduce


def _cast_tuple(value, ndim):
    if isinstance(value, Shape):
        return value._shape
    elif isinstance(value, tuple):
        return value
    elif isinstance(value, int):
        return (value,) * ndim
    else:
        raise ValueError(f"Cannot cast {type(value).__name__} to tuple")


class Shape(object):
    def __init__(self, shape):
        if not isinstance(shape, Iterable):
            self._shape = (shape,)
        else:
            self._shape = tuple(shape)

    def __str__(self):
        return str(self._shape)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Shape(self._shape[idx])
        else:
            return self._shape[idx]

    def __len__(self):
        return len(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    def volume(self):
        return reduce(lambda x, y: x * y, self._shape, 1)

    def sum(self):
        return reduce(lambda x, y: x + y, self._shape, 0)

    def __hash__(self):
        return hash((self.__class__, self._shape))

    def __le__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return len(lh) == len(rh) and lh <= rh

    def __eq__(self, other):
        lh = _cast_tuple(self, self.ndim)
        rh = _cast_tuple(other, self.ndim)
        return lh == rh

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
        return Shape(self._shape[:dim] + self._shape[dim + 1 :])

    def update(self, dim, new_value):
        return self.replace(dim, (new_value,))

    def replace(self, dim, new_values):
        if not isinstance(new_values, tuple):
            new_values = tuple(new_values)
        return Shape(self._shape[:dim] + new_values + self._shape[dim + 1 :])

    def insert(self, dim, new_value):
        return Shape(self._shape[:dim] + (new_value,) + self._shape[dim:])

    def map(self, mapping):
        return Shape(tuple(self[mapping[dim]] for dim in range(self.ndim)))

    def strides(self):
        strides = ()
        stride = 1
        for size in reversed(self._shape):
            strides += (stride,)
            stride *= size
        return Shape(reversed(strides))
