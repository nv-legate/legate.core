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

from typing import TYPE_CHECKING, Protocol, Tuple

import numpy as np

from . import AffineTransform
from .partition import Replicate, Restriction, Tiling
from .projection import ProjExpr
from .runtime import runtime
from .shape import Shape

if TYPE_CHECKING:
    from . import BufferBuilder
    from .partition import PartitionBase
    from .projection import SymbolicPoint


class NonInvertibleError(Exception):
    pass


Restrictions = Tuple[Restriction, ...]


class TransformProto(Protocol):
    def __repr__(self) -> str:
        return str(self)

    def serialize(self, buf: BufferBuilder) -> None:
        ...

    def adds_fake_dims(self) -> bool:
        ...

    @property
    def convertible(self) -> bool:
        ...

    def invert_color(self, color: Shape) -> Shape:
        ...

    def invert_extent(self, extent: Shape) -> Shape:
        ...

    def invert_point(self, point: Shape) -> Shape:
        ...

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        ...

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        ...

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        ...

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        ...


class Transform(TransformProto, Protocol):
    def invert(self, partition: PartitionBase) -> PartitionBase:
        ...

    def convert(self, partition: PartitionBase) -> PartitionBase:
        ...


class Shift(Transform):
    def __init__(self, dim: int, offset: int) -> None:
        self._dim = dim
        self._offset = offset

    def compute_shape(self, shape: Shape) -> Shape:
        return shape

    def __str__(self) -> str:
        return f"Shift(dim: {self._dim}, offset: {self._offset})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._dim == other._dim and self._offset == other._offset

    def __hash__(self) -> int:
        return hash((type(self), self._dim, self._offset))

    def adds_fake_dims(self) -> bool:
        return False

    @property
    def convertible(self) -> bool:
        return True

    def invert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            offset = partition.offset[self._dim] - self._offset
            assert partition.color_shape is not None
            return Tiling(
                partition.tile_shape,
                partition.color_shape,
                partition.offset.update(self._dim, offset),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def invert_color(self, color: Shape) -> Shape:
        return color

    def invert_extent(self, extent: Shape) -> Shape:
        return extent

    def invert_point(self, point: Shape) -> Shape:
        return point.update(self._dim, point[self._dim] - self._offset)

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        return dims

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return restrictions

    def convert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            offset = partition.offset[self._dim] + self._offset
            assert partition.color_shape is not None
            return Tiling(
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

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return restrictions

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        result = AffineTransform(ndim, ndim, True)
        result.offset[self._dim] = -self._offset
        return result

    def serialize(self, buf: BufferBuilder) -> None:
        code = runtime.get_transform_code(self.__class__.__name__)
        buf.pack_32bit_int(code)
        buf.pack_32bit_int(self._dim)
        buf.pack_64bit_int(self._offset)


class Promote(Transform):
    def __init__(self, extra_dim: int, dim_size: int) -> None:
        self._extra_dim = extra_dim
        self._dim_size = dim_size

    def compute_shape(self, shape: Shape) -> Shape:
        return shape.insert(self._extra_dim, self._dim_size)

    def __str__(self) -> str:
        return f"Promote(dim: {self._extra_dim}, size: {self._dim_size})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (
            self._extra_dim == other._extra_dim
            and self._dim_size == other._dim_size
        )

    def __hash__(self) -> int:
        return hash((type(self), self._extra_dim, self._dim_size))

    def adds_fake_dims(self) -> bool:
        return self._dim_size > 1

    @property
    def convertible(self) -> bool:
        return True

    def invert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            return Tiling(
                partition.tile_shape.drop(self._extra_dim),
                partition.color_shape.drop(self._extra_dim),
                partition.offset.drop(self._extra_dim),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def invert_color(self, color: Shape) -> Shape:
        return color.drop(self._extra_dim)

    def invert_extent(self, extent: Shape) -> Shape:
        return extent.drop(self._extra_dim)

    def invert_point(self, point: Shape) -> Shape:
        return point.drop(self._extra_dim)

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        return dims[: self._extra_dim] + dims[self._extra_dim + 1 :]

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        left = restrictions[: self._extra_dim]
        right = restrictions[self._extra_dim + 1 :]
        return left + right

    def convert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            return Tiling(
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

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        left = restrictions[: self._extra_dim]
        right = restrictions[self._extra_dim :]
        new = (Restriction.AVOIDED,)
        return left + new + right

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        parent_ndim = ndim - 1
        result = AffineTransform(parent_ndim, ndim, False)
        parent_dim = 0
        for child_dim in range(ndim):
            if child_dim != self._extra_dim:
                result.trans[parent_dim, child_dim] = 1
                parent_dim += 1
        return result

    def serialize(self, buf: BufferBuilder) -> None:
        code = runtime.get_transform_code(self.__class__.__name__)
        buf.pack_32bit_int(code)
        buf.pack_32bit_int(self._extra_dim)
        buf.pack_64bit_int(self._dim_size)


class Project(Transform):
    def __init__(self, dim: int, index: int) -> None:
        self._dim = dim
        self._index = index

    def compute_shape(self, shape: Shape) -> Shape:
        return shape.drop(self._dim)

    def __str__(self) -> str:
        return f"Project(dim: {self._dim}, index: {self._index})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._dim == other._dim and self._index == other._index

    def __hash__(self) -> int:
        return hash((type(self), self._dim, self._index))

    def adds_fake_dims(self) -> bool:
        return False

    @property
    def convertible(self) -> bool:
        return True

    def invert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            return Tiling(
                partition.tile_shape.insert(self._dim, 1),
                partition.color_shape.insert(self._dim, 1),
                partition.offset.insert(self._dim, self._index),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def invert_color(self, color: Shape) -> Shape:
        return color.insert(self._dim, 0)

    def invert_extent(self, extent: Shape) -> Shape:
        return extent.insert(self._dim, 1)

    def invert_point(self, point: Shape) -> Shape:
        return point.insert(self._dim, self._index)

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        return (
            dims[: self._dim]
            + (ProjExpr(dim=-1, weight=0),)
            + dims[self._dim :]
        )

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        left = restrictions[: self._dim]
        right = restrictions[self._dim :]
        new = (Restriction.UNRESTRICTED,)
        return left + new + right

    def convert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            return Tiling(
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

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return restrictions[: self._dim] + restrictions[self._dim + 1 :]

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        parent_ndim = ndim + 1
        if ndim == 0:
            return AffineTransform(parent_ndim, parent_ndim, False)
        result = AffineTransform(parent_ndim, ndim, False)
        result.offset[self._dim] = self._index
        child_dim = 0
        for parent_dim in range(parent_ndim):
            if parent_dim != self._dim:
                result.trans[parent_dim, child_dim] = 1
                child_dim += 1
        return result

    def serialize(self, buf: BufferBuilder) -> None:
        code = runtime.get_transform_code(self.__class__.__name__)
        buf.pack_32bit_int(code)
        buf.pack_32bit_int(self._dim)
        buf.pack_64bit_int(self._index)


class Transpose(Transform):
    def __init__(self, axes: tuple[int, ...]) -> None:
        self._axes = axes
        self._inverse = tuple(np.argsort(self._axes))

    def compute_shape(self, shape: Shape) -> Shape:
        new_shape = Shape(tuple(shape[dim] for dim in self._axes))
        return new_shape

    def __str__(self) -> str:
        return f"Transpose(axes: {self._axes})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._axes == other._axes

    def __hash__(self) -> int:
        return hash((type(self), tuple(self._axes)))

    def adds_fake_dims(self) -> bool:
        return False

    @property
    def convertible(self) -> bool:
        return True

    def invert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            return Tiling(
                partition.tile_shape.map(self._inverse),
                partition.color_shape.map(self._inverse),
                partition.offset.map(self._inverse),
            )
        else:
            raise ValueError(
                f"Unsupported partition: {type(partition).__name__}"
            )

    def invert_color(self, color: Shape) -> Shape:
        return color.map(self._inverse)

    def invert_extent(self, extent: Shape) -> Shape:
        return extent.map(self._inverse)

    def invert_point(self, point: Shape) -> Shape:
        return point.map(self._inverse)

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        return tuple(dims[idx] for idx in self._inverse)

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return tuple(restrictions[idx] for idx in self._inverse)

    def convert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            return Tiling(
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

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return tuple(restrictions[idx] for idx in self._axes)

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        result = AffineTransform(ndim, ndim, False)
        for dim in range(ndim):
            result.trans[self._axes[dim], dim] = 1
        return result

    def serialize(self, buf: BufferBuilder) -> None:
        code = runtime.get_transform_code(self.__class__.__name__)
        buf.pack_32bit_int(code)
        buf.pack_32bit_uint(len(self._axes))
        for axis in self._axes:
            buf.pack_32bit_int(axis)


class Delinearize(Transform):
    def __init__(self, dim: int, shape: Shape) -> None:
        self._dim = dim
        self._shape = Shape(shape)
        self._strides = self._shape.strides()

    def compute_shape(self, shape: Shape) -> Shape:
        return shape.replace(self._dim, self._shape)

    def __str__(self) -> str:
        return f"Delinearize(dim: {self._dim}, shape: {self._shape})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._dim == other._dim and self._shape == other._shape

    def __hash__(self) -> int:
        return hash((type(self), self._shape, self._strides))

    def adds_fake_dims(self) -> bool:
        return False

    @property
    def convertible(self) -> bool:
        return False

    def invert(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Tiling):
            assert partition.color_shape is not None
            color_shape = partition.color_shape[
                self._dim + 1 : self._dim + self._shape.ndim
            ]
            offset = partition.offset[
                self._dim + 1 : self._dim + self._shape.ndim
            ]

            if color_shape.volume() == 1 and offset.sum() == 0:
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

    def invert_color(self, color: Shape) -> Shape:
        raise NonInvertibleError()

    def invert_extent(self, extent: Shape) -> Shape:
        raise NonInvertibleError()

    def invert_point(self, point: Shape) -> Shape:
        raise NonInvertibleError()

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        left = dims[: self._dim + 1]
        right = dims[self._dim + self._shape.ndim :]
        return left + right

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        left = restrictions[: self._dim + 1]
        right = restrictions[self._dim + self._shape.ndim :]
        return left + right

    def convert(self, partition: PartitionBase) -> PartitionBase:
        raise NonInvertibleError()

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        left = restrictions[: self._dim]
        right = restrictions[self._dim + 1 :]
        new = (Restriction.UNRESTRICTED,) + (Restriction.RESTRICTED,) * (
            self._shape.ndim - 1
        )
        return left + new + right

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
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

        return result

    def serialize(self, buf: BufferBuilder) -> None:
        code = runtime.get_transform_code(self.__class__.__name__)
        buf.pack_32bit_int(code)
        buf.pack_32bit_int(self._dim)
        buf.pack_32bit_uint(self._shape.ndim)
        for extent in self._shape:
            buf.pack_64bit_int(extent)


class TransformStackBase(TransformProto, Protocol):
    @property
    def bottom(self) -> bool:
        ...

    def stack(self, transform: Transform) -> TransformStack:
        ...

    def convert_partition(self, partition: PartitionBase) -> PartitionBase:
        ...

    def _invert_partition(self, partition: PartitionBase) -> PartitionBase:
        ...

    def invert_partition(self, partition: PartitionBase) -> PartitionBase:
        ...


class TransformStack(TransformStackBase):
    def __init__(
        self, transform: Transform, parent: TransformStackBase
    ) -> None:
        self._transform = transform
        self._parent = parent

    def __str__(self) -> str:
        return f"{self._transform} >> {self._parent}"

    def __repr__(self) -> str:
        return str(self)

    def adds_fake_dims(self) -> bool:
        return (
            self._transform.adds_fake_dims() or self._parent.adds_fake_dims()
        )

    @property
    def convertible(self) -> bool:
        return self._transform.convertible and self._parent.convertible

    @property
    def bottom(self) -> bool:
        return False

    def invert_color(self, color: Shape) -> Shape:
        return self._parent.invert_color(self._transform.invert_color(color))

    def invert_extent(self, extent: Shape) -> Shape:
        return self._parent.invert_extent(
            self._transform.invert_extent(extent)
        )

    def invert_point(self, point: Shape) -> Shape:
        return self._parent.invert_point(self._transform.invert_point(point))

    def convert_partition(self, partition: PartitionBase) -> PartitionBase:
        return self._transform.convert(
            self._parent.convert_partition(partition)
        )

    def _invert_partition(self, partition: PartitionBase) -> PartitionBase:
        return self._parent._invert_partition(
            self._transform.invert(partition)
        )

    def invert_partition(self, partition: PartitionBase) -> PartitionBase:
        if isinstance(partition, Replicate):
            return partition
        return self._parent._invert_partition(
            self._transform.invert(partition)
        )

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        return self._parent.invert_symbolic_point(
            self._transform.invert_symbolic_point(dims)
        )

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return self._transform.convert_restrictions(
            self._parent.convert_restrictions(restrictions)
        )

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return self._parent.invert_restrictions(
            self._transform.invert_restrictions(restrictions)
        )

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        transform = self._transform.get_inverse_transform(ndim)
        parent = self._parent.get_inverse_transform(transform.M)
        return transform.compose(parent)

    def stack(self, transform: Transform) -> TransformStack:
        return TransformStack(transform, self)

    def serialize(self, buf: BufferBuilder) -> None:
        self._transform.serialize(buf)
        self._parent.serialize(buf)


class IdentityTransform(TransformStackBase):
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "id"

    def __repr__(self) -> str:
        return str(self)

    def adds_fake_dims(self) -> bool:
        return False

    @property
    def convertible(self) -> bool:
        return True

    @property
    def bottom(self) -> bool:
        return True

    def invert_color(self, color: Shape) -> Shape:
        return color

    def invert_extent(self, extent: Shape) -> Shape:
        return extent

    def invert_point(self, point: Shape) -> Shape:
        return point

    def convert_partition(self, partition: PartitionBase) -> PartitionBase:
        return partition

    def _invert_partition(self, partition: PartitionBase) -> PartitionBase:
        return partition

    def invert_partition(self, partition: PartitionBase) -> PartitionBase:
        return partition

    def invert_symbolic_point(self, dims: SymbolicPoint) -> SymbolicPoint:
        return dims

    def convert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return restrictions

    def invert_restrictions(self, restrictions: Restrictions) -> Restrictions:
        return restrictions

    def get_inverse_transform(self, ndim: int) -> AffineTransform:
        return AffineTransform(ndim, ndim, True)

    def stack(self, transform: Transform) -> TransformStack:
        return TransformStack(transform, self)

    def serialize(self, buf: BufferBuilder) -> None:
        buf.pack_32bit_int(-1)


identity = IdentityTransform()
