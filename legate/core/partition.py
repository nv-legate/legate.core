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


class NoPartition(object):
    @property
    def color_shape(self):
        return None

    def is_disjoint_for(self, strategy, store):
        return not strategy.parallel

    def get_requirement(self, launch_space, store):
        return Broadcast()

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, NoPartition)

    def __str__(self):
        return "NoPartition"

    def __repr__(self):
        return str(self)

    def satisfies_restriction(self, restrictions):
        return True

    def translate(self, offset):
        return self


class Interval(object):
    def __init__(self, lo, extent):
        self._lo = lo
        self._hi = lo + extent

    def overlaps(self, other):
        return not (other._hi <= self._lo or self._hi <= other._lo)


class Tiling(object):
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

    def overlaps(self, other):
        assert self.tile_shape.ndim == other.tile_shape.ndim
        assert self.color_shape.volume() == 1
        assert other.color_shape.volume() == 1

        for dim in range(self.tile_shape.ndim):
            my_interval = Interval(self.offset[dim], self.tile_shape[dim])
            other_interval = Interval(other.offset[dim], other.tile_shape[dim])
            if not my_interval.overlaps(other_interval):
                return False

        return True

    def satisfies_restriction(self, restrictions):
        for dim, restriction in enumerate(restrictions):
            if (
                restriction == Restriction.RESTRICTED
                and self.color_shape[dim] > 1
            ):
                return False
        return True

    def is_complete_for(self, tile):
        my_lo = self._offset
        my_hi = self._offset + self.tile_shape * self.color_shape

        tile_lo = tile._offset
        tile_hi = tile._offset + tile.tile_shape * tile.color_shape

        return my_lo <= tile_lo and tile_hi <= my_hi

    def is_disjoint_for(self, strategy, store):
        inverted = store.invert_partition(self)
        return inverted.color_shape.volume() == self.color_shape.volume()

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

    def get_requirement(self, launch_space, store):
        part, proj_id = store.find_or_create_partition(self)
        if self.color_shape.ndim != launch_space.ndim:
            assert launch_space.ndim == 1
            proj_id = self._runtime.get_delinearize_functor()
        return Partition(part, proj_id)
