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


from .launcher import Broadcast, Partition
from .legion import (
    IndexPartition,
    PartitionByRestriction,
    Rect,
    Transform,
    legion,
)
from .shape import Shape


class NoPartition(object):
    @property
    def color_shape(self):
        return None

    def get_requirement(self, store):
        return Broadcast()

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, NoPartition)

    def __str__(self):
        return "NoPartition"

    def __repr__(self):
        return str(self)


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

    # Returns true if the tiles in the other partition are all contained in
    # this partition
    def subsumes(self, other):
        if not (
            isinstance(other, Tiling) and self._tile_shape == other._tile_shape
        ):
            return False
        elif not (other._color_shape <= self._color_shape):
            return False

        offset = other._offset - self._offset
        # The difference of the offsets must be a multiple of the tile size
        return (offset % self._tile_shape).sum() == 0

    def is_complete_for(self, shape):
        if self._offset.sum() > 0:
            return False
        covered = self._tile_shape * self._color_shape
        return covered >= shape

    def construct(self, region, shape, complete=None, inverse_transform=None):
        tile_shape = self._tile_shape
        transform = Transform(tile_shape.ndim, tile_shape.ndim)
        for idx, size in enumerate(tile_shape):
            transform.trans[idx, idx] = size

        lo = Shape((0,) * tile_shape.ndim) + self._offset
        hi = self._tile_shape - 1 + self._offset

        if inverse_transform is not None:
            inverse = Transform(*inverse_transform.trans.shape)
            inverse.trans = inverse_transform.trans
            transform = transform.compose(inverse)
            lo = inverse_transform.apply(lo)
            hi = inverse_transform.apply(hi)

        extent = Rect(hi, lo, exclusive=False)

        color_space = self._runtime.find_or_create_index_space(
            self.color_shape
        )
        functor = PartitionByRestriction(transform, extent)
        if complete is None:
            complete = self.is_complete_for(shape)
        index_partition = IndexPartition(
            self._runtime.legion_context,
            self._runtime.legion_runtime,
            region.index_space,
            color_space,
            functor,
            kind=legion.LEGION_COMPUTE_KIND,
            keep=True,  # export this partition functor to other libraries
        )
        return region.get_child(index_partition)

    def get_requirement(self, store):
        return Partition(store.find_or_create_partition(self))
