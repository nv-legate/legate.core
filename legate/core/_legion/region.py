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

import weakref
from typing import TYPE_CHECKING, Any, Optional, Union

from .. import ffi, legion
from .partition import IndexPartition, Partition
from .pending import _pending_deletions, _pending_unordered
from .space import IndexSpace
from .util import Dispatchable, FieldID, dispatch

if TYPE_CHECKING:
    from . import FieldListLike, FieldSpace


class Region:

    handle: Any

    def __init__(
        self,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        index_space: IndexSpace,
        field_space: FieldSpace,
        handle: Optional[Any] = None,
        parent: Optional[Partition] = None,
        owned: bool = True,
    ) -> None:
        """
        A Region wraps a `legion_logical_region_t` in the Legion C API.
        A logical region describes a dataframe-like representation of program
        data with the rows described by an IndexSpace and the columns described
        by a FieldSpace.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        index_space : IndexSpace
            The index space for this logical region
        field_space : FieldSpace
            The field space for this logical region
        handle : legion_logical_region_t
            Created handle for a logical region from a Legion C API call
        parent : Partition
            Parent logical partition for this logical region, if any
        owned : bool
            Whether this object owns the handle for this region and
            can delete the region when the object is collected
        """
        if (
            parent is not None
            and index_space.parent is not parent.index_partition
        ):
            raise ValueError("Illegal index space for parent")
        self.context = context
        self.runtime = runtime
        self.index_space = index_space
        self.field_space = field_space
        self.parent = parent
        if handle is None:
            if parent is None:
                # Create a new logical region
                handle = legion.legion_logical_region_create(
                    runtime,
                    context,
                    index_space.handle,
                    field_space.handle,
                    True,
                )
            else:
                handle = legion.legion_logical_partition_get_subregion(
                    runtime, parent.handle, index_space.handle
                )
        self.handle = handle
        self.owned = owned
        # Make this a WeakValueDicitionary so that entries can be
        # removed once the partitions are deleted
        self.children: Any = weakref.WeakValueDictionary()

    def __del__(self) -> None:
        if self.owned and self.parent is None:
            self.destroy(unordered=True)

    def same_handle(self, other: Region) -> bool:
        return (
            type(self) == type(other)
            and self.handle.tree_id == other.handle.tree_id
            and self.handle.index_space.id == other.handle.index_space.id
            and self.handle.field_space.id == other.handle.field_space.id
        )

    def __str__(self) -> str:
        return (
            f"Region("
            f"tid: {self.handle.tree_id}, "
            f"is: {self.handle.index_space.id}, "
            f"fs: {self.handle.field_space.id})"
        )

    def destroy(self, unordered: bool = False) -> None:
        """
        Force deletion of this Region regardless of ownership

        Parameters
        ----------
        unordered : bool
            Whether this request is coming directly from the task or through
            and unordered channel such as a garbage collection
        """
        if self.parent is not None:
            raise RuntimeError("Only root Region objects can be destroyed")
        if not self.owned:
            return
        self.owned = False
        # Check to see if we're still in the context of our task
        # If not we need to leak this region
        if self.context not in _pending_unordered:
            return
        if unordered:
            _pending_unordered[self.context].append((self.handle, type(self)))
        else:
            legion.legion_logical_region_destroy_unordered(
                self.runtime, self.context, self.handle, False
            )

    def get_child(self, index_partition: IndexPartition) -> Partition:
        """
        Find the Partition object that corresponds to the corresponding
        IndexPartition object for the IndexSpace of this Region.
        """
        if index_partition in self.children:
            return self.children[index_partition]
        child = Partition(self.context, self.runtime, index_partition, self)
        self.children[index_partition] = child
        return child

    def get_root(self) -> Region:
        """
        Return the root Region in this tree.
        """
        if self.parent is not None:
            return self.parent.get_root()
        return self


class OutputRegion:
    field_space: Union[FieldSpace, None]
    region: Optional[Region]
    _logical_handle: Any

    def __init__(
        self,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        field_space: Optional[FieldSpace] = None,
        fields: Optional[FieldListLike] = None,
        ndim: Optional[int] = None,
        global_indexing: bool = True,
        existing: Optional[Union[Region, Partition]] = None,
        flags: Optional[int] = None,
        proj: Optional[int] = None,
        parent: Optional[Region] = None,
        coherence: int = legion.LEGION_EXCLUSIVE,
        tag: int = 0,
    ) -> None:
        """
        An OutputRegion creates a name for a logical region that will be
        produced as an output from executing a/an (index) task. The bounds of
        the index space for this region will not be immediately known, but
        users can still use the names of the logical region and index space
        throughout the program without needing to block to wait for the size to
        resolve.

        Parameters
        ----------
        context : legion_context_t
            Context for the enclosing parent task
        runtime : legion_runtime_t
            Handle for the Legion runtime
        field_space : FieldSpace
            The field space to use for the creation of a new logical region
        fields : int or FieldID or List[int] or List[FieldID]
            The fields that will be created in the output region
        global_indexing : bool
            Whether or not the runtime should use a global indexing approach
            to compute the new index space for the output logical region.
            If set to true, the output space will be a 1-D space with a
            prefix sum over all the subregions. If set to false then the
            output index space will be 2-D reflecting the output size for
            each of the different point tasks contributing.
        existing : Region or Partition
            An optional existing region or partition for which the output
            will be a new instance
        flags : int
            Flags to attach to the region requirement for the output region
        proj : int
            The ID of a projection function to use for projection
        parent : Region
            The parent logical region from which privileges are derived
        coherence : int
            The coherence mode for accessing the logical region
        tag : int
            The tag to attach to the region requirement to provide context
        """
        self.context = context
        self.runtime = runtime
        self.fields: set[Union[int, FieldID]] = set()
        self.ndim = 1 if ndim is None else ndim

        if field_space is not None:
            if existing is not None:
                raise ValueError(
                    "'existing' cannot be set if 'field_space' is"
                )
            self.field_space = field_space
            self.region = None
            self.partition = None
            self.handle = legion.legion_output_requirement_create(
                field_space.handle, ffi.NULL, 0, self.ndim, global_indexing
            )
        elif existing is not None:
            if isinstance(existing, Region):
                self.field_space = None
                self.region = existing
                self.partition = None
                if proj is None:
                    req = (
                        legion.legion_region_requirement_create_logical_region(
                            existing.handle,
                            legion.LEGION_WRITE_DISCARD,
                            coherence,
                            existing.get_root().handle
                            if parent is None
                            else parent.handle,
                            tag,
                            False,
                        )
                    )
                else:
                    req = legion.legion_region_requirement_create_logical_region_projection(  # noqa: E501
                        existing.handle,
                        proj,
                        legion.LEGION_WRITE_DISCARD,
                        coherence,
                        existing.get_root().handle
                        if parent is None
                        else parent.handle,
                        tag,
                        False,
                    )
                if flags is not None:
                    legion.legion_region_requirement_add_flags(req, flags)
                self.handle = (
                    legion.legion_output_requirement_create_region_requirement(
                        req
                    )
                )
                legion.legion_region_requirement_destroy(req)
            elif isinstance(existing, Partition):
                self.field_space = None
                self.region = existing.parent
                self.partition = existing
                if proj is None:
                    raise ValueError(
                        "'proj' must be set for partition outputs"
                    )
                req = (
                    legion.legion_region_requirement_create_logical_partition(
                        existing.handle,
                        proj,
                        legion.LEGION_WRITE_DISCARD,
                        coherence,
                        existing.get_root().handle
                        if parent is None
                        else parent.handle,
                        tag,
                        False,
                    )
                )
                if flags is not None:
                    legion.legion_region_requirement_add_flags(req, flags)
                self.handle = (
                    legion.legion_output_requirement_create_region_requirement(
                        req
                    )
                )
                legion.legion_region_requirement_destroy(req)
            else:
                raise TypeError("'existing' must be a Region or Partition")
        else:
            raise ValueError(
                "'existing' must be set if 'field_space' is not set"
            )

        if fields is not None:
            if isinstance(fields, list):
                for field in fields:
                    self.add_field(field)
            else:
                self.add_field(fields)

    def __del__(self) -> None:
        self.destroy(unordered=True)

    def destroy(self, unordered: bool) -> None:
        """
        Eagerly destroy this OutputRegion before the garbage collector does
        It is illegal to use the OutputRegion after this call

        Parameters
        ----------
        unordered : bool
            Whether this OutputRegion is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_future_destroy(self.handle)
        self.handle = None

    def add_field(
        self, field: Union[int, FieldID], instance: bool = True
    ) -> None:
        """
        Add a field to this output region

        Parameters
        ----------
        field : int or FieldID
            field to add to the output region
        instance : bool
            whether this should be added to the instance vector of
            the region requirement for the output region
        """
        if field not in self.fields:
            self.fields.add(field)
            legion.legion_output_requirement_add_field(
                self.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                instance,
            )

    def get_region(self, owned: bool = True) -> Region:
        """
        Parameters
        ----------
        owned : bool
            Whether to take ownership of the created
            IndexSpace and Region objects

        Returns
        -------
        New logical region created by the output region
        """
        if self.region is None:
            assert self.field_space is not None
            handle = legion.legion_output_requirement_get_parent(self.handle)
            index_space = IndexSpace(
                self.context, self.runtime, handle.index_space, owned=owned
            )
            # Dangle a reference here to make sure we don't accidentally
            # cleanup the handle before the IndexSpace goes away
            index_space._logical_handle = handle
            self.region = Region(
                self.context,
                self.runtime,
                index_space,
                self.field_space,
                handle,
                owned=owned,
            )
        return self.region

    def get_partition(self, owned: bool = True) -> Partition:
        """
        Parameters
        ----------
        owned : bool
            Whether to take ownership of the created IndexPartition object

        Returns
        -------
        New logical partition created by the output region
        """
        if self.partition is None:
            assert self.field_space is not None
            context = self.context
            runtime = self.runtime
            parent = self.get_region()
            partition_handle = legion.legion_output_requirement_get_partition(
                self.handle
            )
            color_space = IndexSpace(
                context,
                runtime,
                legion.legion_index_partition_get_color_space(
                    runtime,
                    partition_handle.index_partition,
                ),
                owned=False,
            )
            index_partition = IndexPartition(
                context,
                runtime,
                parent.index_space,
                color_space,
                handle=partition_handle.index_partition,
                owned=owned,
            )
            # Careful here, need to dangle a reference to the partition_handle
            # off the index_partition because its handle references it
            index_partition._logical_handle = partition_handle
            self.partition = Partition(
                context, runtime, index_partition, parent, partition_handle
            )
        return self.partition


class PhysicalRegion(Dispatchable[None]):
    def __init__(self, handle: Any, region: Region) -> None:
        """
        A PhysicalRegion object represents an actual mapping of a logical
        region to a physical allocation in memory and its associated layout.
        PhysicalRegion objects can be both mapped and unmapped. A mapped
        PhysicalRegion contains the most recent valid copy of the logical
        region's data, whereas an unmapped PhysicalRegion can contains a
        stale copy of the data.

        Parameters
        ----------
        handle : legion_physical_region_t
            The handle for a physical region that this object will own
        region : Region
            The logical region for this physical region
        """
        self.handle = handle
        self.region = region

    def __del__(self) -> None:
        self.destroy(unordered=True)

    def destroy(self, unordered: bool) -> None:
        """
        Eagerly destroy this PhysicalRegion before the garbage collector does
        It is illegal to use the PhysicalRegion after this call

        Parameters
        ----------
        unordered : bool
            Whether this PhysicalRegion is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_physical_region_destroy(self.handle)
        self.handle = None

    def is_mapped(self) -> bool:
        """
        Returns
        -------
        bool indicating if this PhysicalRegion is currently mapped
        """
        # It's possible that this method is invoked on an already GC'd
        # PhysicalRegion object, due to the Python GC not being topologically
        # ordered. We ignore the call in this case.
        if self.handle is None:
            return False
        return legion.legion_physical_region_is_mapped(self.handle)

    def wait_until_valid(self) -> None:
        """
        Block waiting until the data in this physical region
        to be ready to access
        """
        legion.legion_physical_region_wait_until_valid(self.handle)

    @dispatch
    def remap(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
    ) -> None:
        """
        Remap this physical region so that it contains a valid copy of the
        data for the logical region that it represents
        """
        legion.legion_runtime_remap_region(runtime, context, self.handle)

    # Launching one of these means remapping it
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        self.remap(runtime, context)

    def unmap(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        unordered: bool = False,
    ) -> None:
        """
        Unmap this physical region from the current logical region
        If 'unordered=True' you must call legate_task_progress to
        guarantee that this unmapping is finished
        """
        # It's possible that this method is invoked on an already GC'd
        # PhysicalRegion object, due to the Python GC not being topologically
        # ordered. We ignore the call in this case.
        if self.handle is None:
            return
        # Check to see if we're still inside the context of the task
        # If not then we just ignore this unmap, it will be done anyway
        if context not in _pending_unordered:
            return
        if unordered:
            _pending_unordered[context].append((self, type(self)))
        else:
            legion.legion_runtime_unmap_region(runtime, context, self.handle)
