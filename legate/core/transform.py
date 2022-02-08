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

import numpy as np

from .legion import AffineTransform
from .partition import Replicate, Restriction, Tiling
from .projection import CoordinateSym
from .shape import Shape


class NonInvertibleError(Exception):
    pass


class Transform(object):
    def __repr__(self):
        return str(self)

    def serialize(self, buf):
        code = self._runtime.get_transform_code(self.__class__.__name__)
        buf.pack_32bit_int(code)


class Shift(Transform):
    def __init__(self, runtime, dim, offset):
        self._runtime = runtime
        self._dim = dim
        self._offset = offset

    def compute_shape(self, shape):
        return shape

    def __str__(self):
        return f"Shift(dim: {self._dim}, offset: {self._offset})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._dim == other._dim and self._offset == other._offset

    def __hash__(self):
        return hash((type(self), self._dim, self._offset))

    def adds_fake_dims(self):
        return False

    @property
    def convertible(self):
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

    def invert_color(self, color):
        return color

    def invert_extent(self, extent):
        return extent

    def invert_point(self, point):
        return point.update(self._dim, point[self._dim] - self._offset)

    def invert_dimensions(self, dims):
        return dims

    def invert_restrictions(self, restrictions):
        return restrictions

    def convert(self, partition):
        if isinstance(partition, Tiling):
            offset = partition.offset[self._dim] + self._offset
            return Tiling(
                self._runtime,
                partition.tile_shape,
                partition.color_shape,
                partition.offset.update(self._dim, offset),
            )
        elif isinstance(partition, Replicate):
            return partition
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def convert_restrictions(self, restrictions):
        return restrictions

    def get_inverse_transform(self, ndim):
        result = AffineTransform(ndim, ndim, True)
        result.offset[self._dim] = -self._offset
        return result, ndim

    def serialize(self, buf):
        super(Shift, self).serialize(buf)
        buf.pack_32bit_int(self._dim)
        buf.pack_64bit_int(self._offset)


class Promote(Transform):
    def __init__(self, runtime, extra_dim, dim_size):
        self._runtime = runtime
        self._extra_dim = extra_dim
        self._dim_size = dim_size

    def compute_shape(self, shape):
        return shape.insert(self._extra_dim, self._dim_size)

    def __str__(self):
        return f"Promote(dim: {self._extra_dim}, size: {self._dim_size})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._extra_dim == other._extra_dim
            and self._dim_size == other._dim_size
        )

    def __hash__(self):
        return hash((type(self), self._extra_dim, self._dim_size))

    def adds_fake_dims(self):
        return True

    @property
    def convertible(self):
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

    def invert_color(self, color):
        return color.drop(self._extra_dim)

    def invert_extent(self, extent):
        return extent.drop(self._extra_dim)

    def invert_point(self, point):
        return point.drop(self._extra_dim)

    def invert_dimensions(self, dims):
        return dims[: self._extra_dim] + dims[self._extra_dim + 1 :]

    def invert_restrictions(self, restrictions):
        left = restrictions[: self._extra_dim]
        right = restrictions[self._extra_dim + 1 :]
        return left + right

    def convert(self, partition):
        if isinstance(partition, Tiling):
            return Tiling(
                self._runtime,
                partition.tile_shape.insert(self._extra_dim, self._dim_size),
                partition.color_shape.insert(self._extra_dim, 1),
                partition.offset.insert(self._extra_dim, 0),
            )
        elif isinstance(partition, Replicate):
            return partition
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def convert_restrictions(self, restrictions):
        left = restrictions[: self._extra_dim]
        right = restrictions[self._extra_dim :]
        new = (Restriction.AVOIDED,)
        return left + new + right

    def get_inverse_transform(self, ndim):
        parent_ndim = ndim - 1
        result = AffineTransform(parent_ndim, ndim, False)
        parent_dim = 0
        for child_dim in range(ndim):
            if child_dim != self._extra_dim:
                result.trans[parent_dim, child_dim] = 1
                parent_dim += 1
        return result, parent_ndim

    def serialize(self, buf):
        super(Promote, self).serialize(buf)
        buf.pack_32bit_int(self._extra_dim)
        buf.pack_64bit_int(self._dim_size)


class Project(Transform):
    def __init__(self, runtime, dim, index):
        self._runtime = runtime
        self._dim = dim
        self._index = index

    def compute_shape(self, shape):
        return shape.drop(self._dim)

    def __str__(self):
        return f"Project(dim: {self._dim}, index: {self._index})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._dim == other._dim and self._index == other._index

    def __hash__(self):
        return hash((type(self), self._dim, self._index))

    def adds_fake_dims(self):
        return False

    @property
    def convertible(self):
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

    def invert_color(self, color):
        return color.insert(self._dim, 0)

    def invert_extent(self, extent):
        return extent.insert(self._dim, 1)

    def invert_point(self, point):
        return point.insert(self._dim, self._index)

    def invert_dimensions(self, dims):
        return dims[: self._dim] + (CoordinateSym(-1),) + dims[self._dim :]

    def invert_restrictions(self, restrictions):
        left = restrictions[: self._dim]
        right = restrictions[self._dim :]
        new = (Restriction.UNRESTRICTED,)
        return left + new + right

    def convert(self, partition):
        if isinstance(partition, Tiling):
            return Tiling(
                self._runtime,
                partition.tile_shape.drop(self._dim),
                partition.color_shape.drop(self._dim),
                partition.offset.drop(self._dim),
            )
        elif isinstance(partition, Replicate):
            return partition
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def convert_restrictions(self, restrictions):
        return restrictions[: self._dim] + restrictions[self._dim + 1 :]

    def get_inverse_transform(self, ndim):
        parent_ndim = ndim + 1
        result = AffineTransform(parent_ndim, ndim, False)
        result.offset[self._dim] = self._index
        child_dim = 0
        for parent_dim in range(parent_ndim):
            if parent_dim != self._dim:
                result.trans[parent_dim, child_dim] = 1
                child_dim += 1
        return result, parent_ndim

    def serialize(self, buf):
        super(Project, self).serialize(buf)
        buf.pack_32bit_int(self._dim)
        buf.pack_64bit_int(self._index)


class Transpose(Transform):
    def __init__(self, runtime, axes):
        self._runtime = runtime
        self._axes = axes
        self._inverse = tuple(np.argsort(self._axes))

    def compute_shape(self, shape):
        new_shape = Shape(tuple(shape[dim] for dim in self._axes))
        return new_shape

    def __str__(self):
        return f"Transpose(axes: {self._axes})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._axes == other._axes

    def __hash__(self):
        return hash((type(self), tuple(self._axes)))

    def adds_fake_dims(self):
        return False

    @property
    def convertible(self):
        return True

    def invert(self, partition):
        if isinstance(partition, Tiling):
            return Tiling(
                self._runtime,
                partition.tile_shape.map(self._inverse),
                partition.color_shape.map(self._inverse),
                partition.offset.map(self._inverse),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def invert_color(self, color):
        return color.map(self._inverse)

    def invert_extent(self, extent):
        return extent.map(self._inverse)

    def invert_point(self, point):
        return point.map(self._inverse)

    def invert_dimensions(self, dims):
        return tuple(dims[idx] for idx in self._inverse)

    def invert_restrictions(self, restrictions):
        return tuple(restrictions[idx] for idx in self._inverse)

    def convert(self, partition):
        if isinstance(partition, Tiling):
            return Tiling(
                self._runtime,
                partition.tile_shape.map(self._axes),
                partition.color_shape.map(self._axes),
                partition.offset.map(self._axes),
            )
        elif isinstance(partition, Replicate):
            return partition
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def convert_restrictions(self, restrictions):
        return tuple(restrictions[idx] for idx in self._axes)

    def get_inverse_transform(self, ndim):
        result = AffineTransform(ndim, ndim, False)
        for dim in range(ndim):
            result.trans[self._axes[dim], dim] = 1
        return result, ndim

    def serialize(self, buf):
        super(Transpose, self).serialize(buf)
        buf.pack_32bit_uint(len(self._axes))
        for axis in self._axes:
            buf.pack_32bit_int(axis)


class Delinearize(Transform):
    def __init__(self, runtime, dim, shape):
        self._runtime = runtime
        self._dim = dim
        self._shape = Shape(shape)
        self._strides = self._shape.strides()

    def compute_shape(self, shape):
        return shape.replace(self._dim, self._shape)

    def __str__(self):
        return f"Delinearize(dim: {self._dim}, shape: {self._shape})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._dim == other._dim and self._shape == other._shape

    def __hash__(self):
        return hash((type(self), self._shape, self._strides))

    def adds_fake_dims(self):
        return False

    @property
    def convertible(self):
        return False

    def invert(self, partition):
        if isinstance(partition, Tiling):
            if (
                partition.color_shape[
                    self._dim + 1 : self._dim + self._shape.ndim
                ].volume()
                == 1
                and partition.offset[
                    self._dim + 1 : self._dim + self._shape.ndim
                ].sum()
                == 0
            ):
                new_tile_shape = partition.tile_shape
                new_color_shape = partition.color_shape
                new_offset = partition.offset
                for _ in range(self._shape.ndim):
                    new_tile_shape = new_tile_shape.drop(self._dim)
                    new_color_shape = new_color_shape.drop(self._dim)
                    new_offset = new_offset.drop(self._dim)

                dim_tile_size = (
                    partition.tile_shape[self._dim] * self._strides[0]
                )
                dim_offset = partition.offset[self._dim] * self._strides[0]
                dim_colors = partition.color_shape[self._dim]

                new_tile_shape = new_tile_shape.insert(
                    self._dim, dim_tile_size
                )
                new_color_shape = new_color_shape.insert(self._dim, dim_colors)
                new_offset = new_offset.insert(self._dim, dim_offset)

                return Tiling(
                    self._runtime,
                    new_tile_shape,
                    new_color_shape,
                    new_offset,
                )
            else:
                raise NonInvertibleError()
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def invert_color(self, color):
        raise NonInvertibleError()

    def invert_extent(self, extent):
        raise NonInvertibleError()

    def invert_point(self, point):
        raise NonInvertibleError()

    def invert_dimensions(self, dims):
        left = dims[: self._dim + 1]
        right = dims[self._dim + self._shape.ndim :]
        return left + right

    def invert_restrictions(self, restrictions):
        left = restrictions[: self._dim + 1]
        right = restrictions[self._dim + self._shape.ndim :]
        return left + right

    def convert_restrictions(self, restrictions):
        left = restrictions[: self._dim]
        right = restrictions[self._dim + 1 :]
        new = (Restriction.UNRESTRICTED,) + (Restriction.RESTRICTED,) * (
            self._shape.ndim - 1
        )
        return left + new + right

    def get_inverse_transform(self, ndim):
        assert ndim >= self._strides.ndim
        out_ndim = ndim - self._strides.ndim + 1
        result = AffineTransform(out_ndim, ndim, False)

        in_dim = 0
        for out_dim in range(out_ndim):
            if out_dim == self._dim:
                for dim, stride in enumerate(self._strides):
                    result.trans[out_dim, in_dim + dim] = stride
                in_dim += self._strides.ndim
            else:
                result.trans[out_dim, in_dim] = 1
                in_dim += 1

        return result, out_ndim

    def serialize(self, buf):
        super(Delinearize, self).serialize(buf)
        buf.pack_32bit_int(self._dim)
        buf.pack_32bit_uint(self._shape.ndim)
        for extent in self._shape:
            buf.pack_64bit_int(extent)


class TransformStack(object):
    def __init__(self, transform, parent):
        self._transform = transform
        self._parent = parent

    def __str__(self):
        return f"{self._transform} >> {self._parent}"

    def __repr__(self):
        return str(self)

    def add_fake_dims(self):
        return self._transform.adds_fake_dims() or self._parent.add_fake_dims()

    @property
    def convertible(self):
        return self._transform.convertible and self._parent.convertible

    @property
    def bottom(self):
        return False

    def invert_color(self, color):
        return self._parent.invert_color(self._transform.invert_color(color))

    def invert_extent(self, extent):
        return self._parent.invert_extent(
            self._transform.invert_extent(extent)
        )

    def invert_point(self, point):
        return self._parent.invert_point(self._transform.invert_point(point))

    def convert_partition(self, partition):
        return self._transform.convert(
            self._parent.convert_partition(partition)
        )

    def _invert_partition(self, partition):
        return self._parent._invert_partition(
            self._transform.invert(partition)
        )

    def invert_partition(self, partition):
        if isinstance(partition, Replicate):
            return partition
        return self._parent._invert_partition(
            self._transform.invert(partition)
        )

    def invert_dimensions(self, dims):
        return self._parent.invert_dimensions(
            self._transform.invert_dimensions(dims)
        )

    def convert_restrictions(self, restrictions):
        return self._transform.convert_restrictions(
            self._parent.convert_restrictions(restrictions)
        )

    def invert_restrictions(self, restrictions):
        return self._parent.invert_restrictions(
            self._transform.invert_restrictions(restrictions)
        )

    def get_inverse_transform(self, ndim):
        transform, ndim = self._transform.get_inverse_transform(ndim)
        parent = self._parent.get_inverse_transform(ndim)
        return transform.compose(parent)

    def stack(self, transform):
        return TransformStack(transform, self)

    def serialize(self, buf):
        self._transform.serialize(buf)
        self._parent.serialize(buf)


class IdentityTransform(object):
    def __init__(self):
        pass

    def __str__(self):
        return "id"

    def __repr__(self):
        return str(self)

    def add_fake_dims(self):
        return False

    @property
    def convertible(self):
        return True

    @property
    def bottom(self):
        return True

    def invert_color(self, color):
        return color

    def invert_extent(self, extent):
        return extent

    def invert_point(self, point):
        return point

    def convert_partition(self, partition):
        return partition

    def _invert_partition(self, partition):
        return partition

    def invert_partition(self, partition):
        return partition

    def invert_dimensions(self, dims):
        return dims

    def convert_restrictions(self, restrictions):
        return restrictions

    def invert_restrictions(self, restrictions):
        return restrictions

    def get_inverse_transform(self, ndim):
        return AffineTransform(ndim, ndim, True)

    def stack(self, transform):
        return TransformStack(transform, self)

    def serialize(self, buf):
        buf.pack_32bit_int(-1)
