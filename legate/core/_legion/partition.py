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

from typing import TYPE_CHECKING, Any, Optional

from .. import legion
from .partition_functor import PartitionFunctor
from .pending import _pending_unordered
from .space import IndexSpace

if TYPE_CHECKING:
    from . import Point, Region


class Partition:
    handle: Any
    index_partition: IndexPartition
    parent: Region

    def __init__(
        self,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        index_partition: IndexPartition,
        parent: Region,
        handle: Optional[Any] = None,
    ) -> None:
        """
        A Partition wraps a `legion_logical_partition_t` in the Legion C API.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        index_partition : IndexPartition
            The index partition associated with this logical partition
        parent : Region
            Parent logical region for this logical partition if any
        handle : legion_logical_partition_t
            Created handle for logical partition from a Legion C API call
        """
        if index_partition.parent is not parent.index_space:
            raise ValueError("Illegal index partition for parent")
        self.context = context
        self.runtime = runtime
        self.index_partition = index_partition
        self.parent = parent
        if handle is None:
            handle = legion.legion_logical_partition_create(
                runtime,
                parent.handle,
                index_partition.handle,
            )
        self.handle = handle
        self.children: dict[Point, Region] = dict()

    @property
    def color_space(self) -> IndexSpace:
        return self.index_partition.color_space

    def destroy(self) -> None:
        """
        This method is deprecated and is a no-op
        Partition objects never need to explicitly destroy their handles
        """
        pass

    def get_child(self, point: Point) -> Region:
        """
        Return the child Region associated with the point in the color space.
        """
        from .region import Region  # circular

        if point in self.children:
            return self.children[point]
        child_space = self.index_partition.get_child(point)
        handle = legion.legion_logical_partition_get_logical_subregion(
            self.runtime, self.handle, child_space.handle
        )
        child = Region(
            self.context,
            self.runtime,
            child_space,
            self.parent.field_space,
            handle=handle,
            parent=self,
        )
        self.children[point] = child
        return child

    def get_root(self) -> Region:
        """
        Return the Region at the root of this region tree.
        """
        return self.parent.get_root()


class IndexPartition:
    _logical_handle: Any

    def __init__(
        self,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        functor: Optional[Any] = None,
        handle: Optional[Any] = None,
        kind: int = legion.LEGION_COMPUTE_KIND,
        part_id: int = legion.legion_auto_generate_id(),
        owned: bool = True,
        keep: bool = False,
    ) -> None:
        """
        An IndexPartition wraps a `legion_index_partition_t` in the Legion C
        API. It describes a partitioning of an IndexSpace into a collection of
        child IndexSpace objects. IndexPartition objects can be both disjoint
        and aliased. They can also be both complete and incomplete partitions.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        parent : IndexSpace
            The parent index space for this partition
        color_space : IndexSpace
            The index space that describes this subregions in this partition
        functor : PartitionFunctor
            Any object that implements the PartitionFunctor interface
        handle : legion_index_partition_t
            Created handle for an index partition from a Legion C API call
        kind : legion_partition_kind_t
            The disjointness and completeness properties for the partition
        part_id : int
            The color for the partition in parent index space
        owned : bool
            Whether this object owns the handle for this index space and
            can delete the index space handle when the object is collected
        keep : bool
            Whether to keep a reference to the functor object that
            was used to create this index partition
        """
        self.context = context
        self.runtime = runtime
        self.parent = parent
        self.color_space = color_space
        if handle is None:
            if functor is None:
                raise ValueError(
                    "'functor' must be " "specified if 'handle is None"
                )
            if not isinstance(functor, PartitionFunctor):
                raise TypeError("'functor' must be a 'PartitionFunctor'")
            handle = functor.partition(
                runtime, context, parent, color_space, kind, part_id
            )
            self.functor = functor if keep else None
        elif functor is not None:
            raise ValueError("'functor' must be None if 'handle' is specified")
        else:
            self.functor = None
        self.handle = handle
        self.children: dict[Point, IndexSpace] = dict()
        self.owned = owned
        if owned and not self.parent.owned:
            raise ValueError(
                "IndexPartition can only own its handle if "
                "the parent IndexSpace also owns its handle"
            )
        self.parent.add_child(self)

    def __del__(self) -> None:
        # Record a pending deletion if this task is still executing
        if self.owned and self.parent._can_delete():
            self.destroy(unordered=True, recursive=True)

    def get_child(self, point: Point) -> IndexSpace:
        """
        Find the child IndexSpace assocated with the point in the color space.
        """
        if point in self.children:
            return self.children[point]
        child_handle = legion.legion_index_partition_get_index_subspace_domain_point(  # noqa: E501
            self.runtime, self.handle, point.raw()
        )
        child = IndexSpace(
            self.context,
            self.runtime,
            handle=child_handle,
            parent=self,
            owned=self.owned,
        )
        self.children[point] = child
        return child

    def destroy(self, unordered: bool = False, recursive: bool = True) -> None:
        """
        Force deletion of this IndexPartition regardless of ownership

        Parameters
        ----------
        unordered : bool
            Whether this request is coming directly from the task or through
            and unordered channel such as a garbage collection
        recursive : bool
            Whether to recursively destroy down the index space tree
        """
        if not self.owned:
            return
        self.owned = False
        # Check to see if we're still inside the context of the task
        # If not then we just need to leak this index partition
        if self.context not in _pending_unordered:
            return
        if unordered:
            # See if we have a _logical_handle from an OutputRegion
            # that we need to pass along to keep things alive
            if hasattr(self, "_logical_handle"):
                _pending_unordered[self.context].append(
                    (
                        (
                            self.handle,
                            recursive,
                            getattr(self, "_logical_handle"),
                        ),
                        type(self),
                    )
                )
            else:
                _pending_unordered[self.context].append(
                    ((self.handle, recursive, None), type(self))
                )
        else:
            legion.legion_index_partition_destroy_unordered(
                self.runtime, self.context, self.handle, False, recursive
            )
        if recursive:
            for child in self.children.values():
                child.owned = False

    def get_root(self) -> IndexSpace:
        """
        Return the root IndexSpace in this tree.
        """
        return self.parent.get_root()
