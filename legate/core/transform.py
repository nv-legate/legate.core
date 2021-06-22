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


import numpy as np

from .legion import AffineTransform
from .partition import Tiling
from .solver import Shape


class Transform(object):
    def __repr__(self):
        return str(self)

    def serialize(self, launcher):
        code = self._runtime.get_transform_code(self.__class__.__name__)
        launcher.add_scalar_arg(code, np.int32)


class Shift(Transform):
    def __init__(self, runtime, dim, offset):
        self._runtime = runtime
        self._dim = dim
        self._offset = offset

    def compute_shape(self, shape):
        return shape

    def __str__(self):
        return f"Shift(dim: {self._dim}, slice: {self._offset})"

    @property
    def invertible(self):
        return True

    def invert(self, partition):
        if isinstance(partition, Tiling):
            offset = partition.offset[self._dim] - self._offset
            return Tiling(
                self._runtime,
                partition.tile_shape,
                partition.color_shape,
                partition.offset.update(self._dim, offset),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def get_accessor_transform(self, shape, parent_transform=None):
        result = AffineTransform(shape.ndim, shape.ndim, True)
        result.offset[self._dim] = -self._offset
        if parent_transform is not None:
            result = result.compose(parent_transform)
        return result

    def serialize(self, launcher):
        super(Shift, self).serialize(launcher)
        launcher.add_scalar_arg(self._dim, np.int32)
        launcher.add_scalar_arg(self._offset, np.int64)


class Promote(Transform):
    def __init__(self, runtime, extra_dim, dim_size):
        self._runtime = runtime
        self._extra_dim = extra_dim
        self._dim_size = dim_size

    def compute_shape(self, shape):
        new_shape = Shape(
            shape[: self._extra_dim]
            + (self._dim_size,)
            + shape[self._extra_dim :]
        )
        return new_shape

    def __str__(self):
        return f"Promote(dim: {self._extra_dim}, size: {self._dim_size})"

    @property
    def invertible(self):
        return True

    def invert(self, partition):
        if isinstance(partition, Tiling):
            return Tiling(
                self._runtime,
                partition.tile_shape.drop(self._extra_dim),
                partition.color_shape.drop(self._extra_dim),
                partition.offset.drop(self._extra_dim),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    # Returns the parent partition that coincides a given partition of a child
    def get_accessor_transform(self, shape, parent_transform=None):
        parent_ndim = shape.ndim - 1
        result = AffineTransform(parent_ndim, shape.ndim, False)
        parent_dim = 0
        for child_dim in range(shape.ndim):
            if child_dim != self._extra_dim:
                result.trans[parent_dim, child_dim] = 1
                parent_dim += 1
        if parent_transform is not None:
            result = result.compose(parent_transform)
        return result

    def serialize(self, launcher):
        super(Promote, self).serialize(launcher)
        launcher.add_scalar_arg(self._extra_dim, np.int32)
        launcher.add_scalar_arg(self._dim_size, np.int64)


class Project(Transform):
    def __init__(self, runtime, dim, index):
        self._runtime = runtime
        self._dim = dim
        self._index = index

    def compute_shape(self, shape):
        new_shape = Shape(shape[: self._dim] + shape[self._dim + 1 :])
        return new_shape

    def __str__(self):
        return f"Project(dim: {self._dim}, index: {self._index})"

    @property
    def invertible(self):
        return True

    def invert(self, partition):
        if isinstance(partition, Tiling):
            return Tiling(
                self._runtime,
                partition.tile_shape.insert(self._dim, 1),
                partition.color_shape.insert(self._dim, 1),
                partition.offset.insert(self._dim, self._index),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    # Returns the parent partition that coincides a given partition of a child
    def get_accessor_transform(self, shape, parent_transform=None):
        parent_ndim = shape.ndim + 1
        result = AffineTransform(parent_ndim, shape.ndim, False)
        result.offset[self._dim] = self._index
        child_dim = 0
        for parent_dim in range(shape.ndim):
            if parent_dim != self._dim:
                result.trans[parent_dim, child_dim] = 1
                child_dim += 1
        if parent_transform is not None:
            result = result.compose(parent_transform)
        return result

    def serialize(self, launcher):
        super(Project, self).serialize(launcher)
        launcher.add_scalar_arg(self._dim, np.int32)
        launcher.add_scalar_arg(self._index, np.int64)
