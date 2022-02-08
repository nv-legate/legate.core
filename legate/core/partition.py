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

from enum import IntEnum, unique

from .launcher import Broadcast, Partition
from .legion import (
    IndexPartition,
    PartitionByRestriction,
    Rect,
    Transform,
    legion,
)
from .shape import Shape


@unique
class Restriction(IntEnum):
    RESTRICTED = -2
    AVOIDED = -1
    UNRESTRICTED = 1


class PartitionBase(object):
    pass


class Replicate(PartitionBase):
    @property
    def color_shape(self):
        return None

    @property
    def requirement(self):
        return Broadcast

    def is_complete_for(self, extents, offsets):
        return True

    def is_disjoint_for(self, launch_domain):
        return launch_domain is None

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, Replicate)

    def __str__(self):
        return "Replicate"

    def __repr__(self):
        return str(self)

    def needs_delinearization(self, launch_ndim):
        return False

    def satisfies_restriction(self, restrictions):
        return True

    def translate(self, offset):
        return self

    def construct(self, region, complete=False):
        return None


REPLICATE = Replicate()


class Interval(object):
    def __init__(self, lo, extent):
        self._lo = lo
        self._hi = lo + extent

    def overlaps(self, other):
        return not (other._hi <= self._lo or self._hi <= other._lo)


class Tiling(PartitionBase):
    def __init__(self, runtime, tile_shape, color_shape, offset=None):
        assert len(tile_shape) == len(color_shape)
        if offset is None:
            offset = Shape((0,) * len(tile_shape))
        self._runtime = runtime
        self._tile_shape = tile_shape
        self._color_shape = color_shape
        self._offset = offset
        self._hash = None

    def __eq__(self, other):
        return (
            isinstance(other, Tiling)
            and self._tile_shape == other._tile_shape
            and self._color_shape == other._color_shape
            and self._offset == other._offset
        )

    @property
    def runtime(self):
        return self._runtime

    @property
    def tile_shape(self):
        return self._tile_shape

    @property
    def color_shape(self):
        return self._color_shape

    @property
    def requirement(self):
        return Partition

    @property
    def offset(self):
        return self._offset

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (
                    self.__class__,
                    self._tile_shape,
                    self._color_shape,
                    self._offset,
                )
            )
        return self._hash

    def __str__(self):
        return (
            f"Tiling(tile:{self._tile_shape}, "
            f"color:{self._color_shape}, "
            f"offset:{self._offset})"
        )

    def __repr__(self):
        return str(self)

    def needs_delinearization(self, launch_ndim):
        return launch_ndim != self._color_shape.ndim

    def satisfies_restriction(self, restrictions):
        for dim, restriction in enumerate(restrictions):
            if (
                restriction == Restriction.RESTRICTED
                and self.color_shape[dim] > 1
            ):
                return False
        return True

    def is_complete_for(self, extents, offsets):
        my_lo = self._offset
        my_hi = self._offset + self.tile_shape * self.color_shape

        return my_lo <= offsets and offsets + extents <= my_hi

    def is_disjoint_for(self, launch_domain):
        return launch_domain.get_volume() <= self.color_shape.volume()

    def has_color(self, color):
        return color >= 0 and color < self._color_shape

    def get_subregion_size(self, extents, color):
        lo = self._tile_shape * color + self._offset
        hi = self._tile_shape * (color + 1) + self._offset
        lo = Shape(max(0, coord) for coord in lo)
        hi = Shape(min(max, coord) for (max, coord) in zip(extents, hi))
        return Shape(hi - lo)

    def get_subregion_offsets(self, color):
        return self._tile_shape * color + self._offset

    def translate(self, offset):
        return Tiling(
            self._runtime,
            self._tile_shape,
            self._color_shape,
            self._offset + offset,
        )

    def construct(self, region, complete=False):
        index_space = region.index_space
        index_partition = self._runtime.find_partition(index_space, self)
        if index_partition is None:
            tile_shape = self._tile_shape
            transform = Transform(tile_shape.ndim, tile_shape.ndim)
            for idx, size in enumerate(tile_shape):
                transform.trans[idx, idx] = size

            lo = Shape((0,) * tile_shape.ndim) + self._offset
            hi = self._tile_shape - 1 + self._offset

            extent = Rect(hi, lo, exclusive=False)

            color_space = self._runtime.find_or_create_index_space(
                self.color_shape
            )
            functor = PartitionByRestriction(transform, extent)
            if complete:
                kind = legion.LEGION_DISJOINT_COMPLETE_KIND
            else:
                kind = legion.LEGION_DISJOINT_INCOMPLETE_KIND
            index_partition = IndexPartition(
                self._runtime.legion_context,
                self._runtime.legion_runtime,
                index_space,
                color_space,
                functor,
                kind=kind,
                keep=True,  # export this partition functor to other libraries
            )
            self._runtime.record_partition(index_space, self, index_partition)
        return region.get_child(index_partition)
