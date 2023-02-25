# Copyright 2022 NVIDIA Corporation
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

from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from . import Partition as LegionPartition, Point
    from .store import RegionField


class InlineMappedAllocation:
    """
    This helper class is to tie the lifecycle of the client object to
    the inline mapped allocation
    """

    def __init__(
        self,
        region_field: RegionField,
        shape: tuple[int, ...],
        address: int,
        strides: tuple[int, ...],
    ) -> None:
        self._region_field = region_field
        self._shape = shape
        self._address = address
        self._strides = strides
        self._consumed = False

    def __del__(self) -> None:
        if not self._consumed:
            self._region_field.decrement_inline_mapped_ref_count(
                unordered=True
            )

    def consume(
        self, ctor: Callable[[tuple[int, ...], int, tuple[int, ...]], Any]
    ) -> Any:
        """
        Consumes the allocation. Each allocation can be consumed only once.

        Parameters
        ----------
        ctor : Callback
            Callback that constructs a Python object from the allocation.
            Each callback gets the shape, the physical address, and the strides
            of the allocation, and is supposed to return a Python object
            using the allocation. Leaking the three arguments in some other way
            will lead to an undefined behavior.

        Returns
        -------
        Any
            Python object the callback constructs from the allocation
        """
        if self._consumed:
            raise RuntimeError("Each inline mapping can be consumed only once")
        self._consumed = True
        result = ctor(self._shape, self._address, self._strides)
        self._region_field.register_consumer(result)
        return result


class DistributedAllocation:
    def __init__(
        self,
        partition: LegionPartition,
        shard_local_buffers: dict[Point, memoryview],
    ) -> None:
        """
        Represents a distributed collection of buffers, to be
        collectively attached as sub-regions of the same
        parent region.

        This is a rare case of a data structure that is allowed (and expected)
        to have a different value on different shards; each shard should
        specify a distinct set of resources.

        Parameters
        ----------
        partition : Partition
            The partition to use in the IndexAttach operation
        shard_local_buffers : dict[Point, memoryview]
            Map from color to buffer that should back the sub-region of that
            color. This map will only cover the buffers local to the current
            shard.
        """
        self.partition = partition
        self.shard_local_buffers = shard_local_buffers


Attachable = Union[memoryview, DistributedAllocation]
