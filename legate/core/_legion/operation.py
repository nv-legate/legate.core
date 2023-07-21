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

from typing import TYPE_CHECKING, Any, Optional, Union

from .. import ffi, legion
from .future import Future
from .geometry import Domain, Point
from .partition import Partition
from .pending import _pending_unordered
from .region import PhysicalRegion, Region
from .space import IndexSpace
from .util import Dispatchable, ExternalResources, FieldID, Mappable, dispatch

if TYPE_CHECKING:
    from . import FieldListLike, Rect


class InlineMapping(Dispatchable[PhysicalRegion]):
    def __init__(
        self,
        region: Region,
        fields: FieldListLike,
        read_only: bool = False,
        mapper: int = 0,
        tag: int = 0,
        parent: Optional[Region] = None,
        coherence: int = legion.LEGION_EXCLUSIVE,
        provenance: Optional[str] = None,
    ) -> None:
        """
        An InlineMapping object provides a mechanism for creating a mapped
        PhysicalRegion of a logical region for the local task to directly
        access the data in the logical region. Note that inline mappings
        do block deferred execution and therefore they should be used
        primarily as a productivity feature for loading and storing data
        infrequently. They should never be used in performance critical code.

        Parameters
        ----------
        region : Region
            The logical region to map
        fields : int or FieldID or List[int] or List[FieldID]
            The fields of the logical region to map
        read_only : bool
            Whether the inline mapping will only be reading the data
        mapper : int
            ID of the mapper for managing the mapping of the inline mapping
        tag : int
            Tag to pass to the mapper to provide calling context
        parent : Region
            Parent logical region from which privileges are derived
        coherence : int
            The coherence mode for the inline mapping
        """
        if read_only:
            self.launcher = legion.legion_inline_launcher_create_logical_region(  # noqa: E501
                region.handle,
                legion.LEGION_READ_ONLY,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                0,
                False,
                mapper,
                tag,
            )
        else:
            self.launcher = legion.legion_inline_launcher_create_logical_region(  # noqa: E501
                region.handle,
                legion.LEGION_READ_WRITE,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                0,
                False,
                mapper,
                tag,
            )
        if provenance is not None:
            legion.legion_inline_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self.region = region
        self._launcher = ffi.gc(
            self.launcher, legion.legion_inline_launcher_destroy
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_inline_launcher_add_field(
                self.launcher,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> PhysicalRegion:
        """
        Dispatch the inline mapping to the runtime

        Returns
        -------
        PhysicalRegion
        """
        return PhysicalRegion(
            legion.legion_inline_launcher_execute(
                runtime, context, self.launcher
            ),
            self.region,
        )


class Fill(Dispatchable[None], Mappable):
    def __init__(
        self,
        region: Region,
        parent: Region,
        field: Union[int, FieldID],
        future: Future,
        mapper: int = 0,
        tag: int = 0,
        provenance: Optional[str] = None,
    ) -> None:
        """
        A Fill object provides a mechanism for launching fill operations
        in the Legion C API. A fill operation initializes the data in a
        field of a logical region to a specific value. Importantly this is
        performed lazily so fills incur near-zero cost.

        Parameters
        ----------
        region : Region
            The logical region to be filled
        parent : Region
            The parent logical region where priviles are derived from
        field : FieldID or int
            The name of the field to be filled
        future : Future
            Future describing the value to use for performing the fill
        mapper : int
            ID of the mapper to use for mapping the fill
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        if not isinstance(field, FieldID) and not isinstance(field, int):
            raise TypeError("Fill field must be int or FieldID")
        self.launcher = legion.legion_fill_launcher_create_from_future(
            region.handle,
            parent.handle if parent is not None else parent.get_root().handle,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            future.handle,
            legion.legion_predicate_true(),
            mapper,
            tag,
        )
        if provenance is not None:
            legion.legion_fill_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_fill_launcher_destroy
        )

    def set_point(self, point: Union[Point, Any]) -> None:
        """
        Set the point to describe this fill for sharding
        with control replication

        Parameters
        ----------
        point : Point or legion_domain_point_t
            The point value to associate with this fill
        """
        if isinstance(point, Point):
            legion.legion_fill_launcher_set_point(self.launcher, point.raw())
        else:
            legion.legion_fill_launcher_set_point(self.launcher, point)

    def set_sharding_space(self, space: IndexSpace) -> None:
        """
        Set the sharding space for this individual fill launch

        Parameters
        ----------
        space : IndexSpace
            The index space for use when performing sharding of this fill
        """
        legion.legion_fill_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_mapper_arg(self, data: Any, size: int) -> None:
        legion.legion_fill_launcher_set_mapper_arg(
            self.launcher,
            (ffi.from_buffer(data), size),
        )
        # Hold a reference to the data to prevent collection
        self.data = data

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        """
        Dispatch the fill to the runtime
        """
        legion.legion_fill_launcher_execute(runtime, context, self.launcher)


class IndexFill(Dispatchable[None], Mappable):
    def __init__(
        self,
        partition: Partition,
        proj: int,
        parent: Region,
        field: Union[int, FieldID],
        future: Future,
        mapper: int = 0,
        tag: int = 0,
        space: Optional[Union[IndexSpace, Domain]] = None,
        provenance: Optional[str] = None,
    ) -> None:
        """
        An IndexFill object provides a mechanism for launching index space fill
        operations in the Legion C API. Index fill operations enable many
        subregions in a sub-tree to be filled concurrently and efficiently.  An
        index fill operation initializes the data in the field of each of the
        logical regions targeted by the index fill operation. Importantly, this
        is performed lazily so index fills incur near-zero cost.

        partition : Partition
            The logical partition upper bound from which to project the
            point fills from the index fill
        proj : int
            Projection function ID to describe how to project
            from the upper bound partition
        parent : Region
            The parent logical region where privileges are derived from
        field : FieldID or int
            The name of the field to be filled
        future: Future
            Future describing the value to use for performing the index fill
        mapper : int
            ID of the mapper to use for mapping the index fill
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        space : IndexSpace or Domain (default:None)
            launch space domain
        """
        if not isinstance(field, FieldID) and not isinstance(field, int):
            raise TypeError("Fill field must be int or FieldID")
        if (
            space is not None
            and not isinstance(space, IndexSpace)
            and not isinstance(space, Domain)
        ):
            raise TypeError(
                "IndexFill launch space must be IndexSpace or Domain"
            )
        if space is None:
            self.launcher = legion.legion_index_fill_launcher_create_from_future_with_space(  # noqa: E501
                partition.index_partition.color_space.handle,
                partition.handle,
                parent.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                future.handle,
                proj,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        elif isinstance(space, IndexSpace):
            self.launcher = legion.legion_index_fill_launcher_create_from_future_with_space(  # noqa: E501
                space.handle,
                partition.handle,
                parent.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                future.handle,
                proj,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        else:
            self.launcher = legion.legion_index_fill_launcher_create_from_future_with_domain(  # noqa: E501
                space.domain,
                partition.handle,
                parent.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                future.handle,
                proj,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        if provenance is not None:
            legion.legion_index_fill_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_fill_launcher_destroy
        )

    def set_sharding_space(self, space: IndexSpace) -> None:
        """
        Set the sharding space to use for this index fill launch
        with control replication

        Parameters
        ----------
        space : IndexSpace
            The index space to use as the sharding space
        """
        legion.legion_index_fill_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_mapper_arg(self, data: Any, size: int) -> None:
        legion.legion_index_fill_launcher_set_mapper_arg(
            self.launcher,
            (ffi.from_buffer(data), size),
        )
        # Hold a reference to the data to prevent collection
        self.data = data

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        """
        Dispatch the index fill to the runtime
        """
        legion.legion_index_fill_launcher_execute(
            runtime, context, self.launcher
        )


class Copy(Dispatchable[None], Mappable):
    def __init__(
        self,
        mapper: int = 0,
        tag: int = 0,
        provenance: Optional[str] = None,
    ) -> None:
        """
        A Copy object provides a mechanism for launching explicit
        region-to-region copy operations. Note: you should NOT use
        these for trying to move data between memories! Copy launchers
        should only be used for logically moving data between fields.

        Parameters
        ----------
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_copy_launcher_create(
            legion.legion_predicate_true(), mapper, tag
        )
        if provenance is not None:
            legion.legion_copy_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_copy_launcher_destroy
        )
        self.src_req_index = 0
        self.dst_req_index = 0

    def set_possible_src_indirect_out_of_range(self, flag: bool) -> None:
        """
        For gather indirection copies indicate whether any of the
        source indirection pointers may point out of bounds in the
        source logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether source pointers may point outside the privileges
        """
        legion.legion_copy_launcher_set_possible_src_indirect_out_of_range(
            self._launcher, flag
        )

    def set_possible_dst_indirect_out_of_range(self, flag: bool) -> None:
        """
        For scatter indirection copies indicate whether any of the
        destination indirection pointers may point out of bounds in the
        destination logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether destination pointers may point outside
            the privileges
        """
        legion.legion_copy_launcher_set_possible_dst_indirect_out_of_range(
            self._launcher, flag
        )

    def add_src_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        parent: Optional[Region] = None,
        tag: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a source region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a source of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as sources of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        coherence : int
            The coherence mode for which to access this region
        """
        legion.legion_copy_launcher_add_src_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_READ_ONLY,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_copy_launcher_add_src_field(
                self.launcher,
                self.src_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.src_req_index += 1

    def add_dst_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        parent: Optional[Region] = None,
        tag: int = 0,
        redop: int = 0,
        privilege: int = legion.LEGION_WRITE_DISCARD,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a destination region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a destination of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as destinations of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        redop : int
            Optional reduction operator ID to reduce to the destination fields
        privilege : int
            Optional privilege for the destination. Ignored when `redop` is
            non-zero. WRITE_DISCARD by default.
        coherence : int
            The coherence mode for which to access this region
        """
        if redop == 0:
            legion.legion_copy_launcher_add_dst_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                region.handle,
                privilege,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                tag,
                False,
            )
        else:
            legion.legion_copy_launcher_add_dst_region_requirement_logical_region_reduction(  # noqa: E501
                self.launcher,
                region.handle,
                redop,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                tag,
                False,
            )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_copy_launcher_add_dst_field(
                self.launcher,
                self.dst_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.dst_req_index += 1

    def add_src_indirect_requirement(
        self,
        region: Region,
        field: Union[int, FieldID],
        parent: Optional[Region] = None,
        tag: int = 0,
        is_range: bool = False,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a source indirection region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a source indirection of the copy
        field : int or FieldID
            The ID of the field to serve as a source indirection of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        legion.legion_copy_launcher_add_src_indirect_region_requirement_logical_region(  # noqa: E501
            self.launcher,
            region.handle,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            is_range,
            False,
        )

    def add_dst_indirect_requirement(
        self,
        region: Region,
        field: Union[int, FieldID],
        parent: Optional[Region] = None,
        tag: int = 0,
        is_range: bool = False,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a destination indirection region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a destination
            indirection of the copy
        field : int or FieldID
            The ID of the field to serve as a destination
            indirection field of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        legion.legion_copy_launcher_add_dst_indirect_region_requirement_logical_region(  # noqa: E501
            self.launcher,
            region.handle,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            is_range,
            False,
        )

    def set_point(self, point: Union[Point, Any]) -> None:
        """
        Set the point to describe this copy for sharding
        with control replication

        Parameters
        ----------
        point : Point or legion_domain_point_t
            The point value to associate with this copy
        """
        if isinstance(point, Point):
            legion.legion_copy_launcher_set_point(self.launcher, point.raw())
        else:
            legion.legion_copy_launcher_set_point(self.launcher, point)

    def set_sharding_space(self, space: IndexSpace) -> None:
        """
        Set the sharding space for this individual copy launch

        Parameters
        ----------
        space : IndexSpace
            The index space for use when performing sharding of this copy
        """
        legion.legion_copy_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_mapper_arg(self, data: Any, size: int) -> None:
        legion.legion_copy_launcher_set_mapper_arg(
            self.launcher,
            (ffi.from_buffer(data), size),
        )
        # Hold a reference to the data to prevent collection
        self.data = data

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        """
        Dispatch the copy operation to the runtime
        """
        if self.src_req_index != self.dst_req_index:
            raise RuntimeError(
                "Number of source and destination requirements "
                + "must match for copies"
            )
        legion.legion_copy_launcher_execute(runtime, context, self.launcher)


class IndexCopy(Dispatchable[None], Mappable):
    def __init__(
        self,
        domain: Rect,
        mapper: int = 0,
        tag: int = 0,
        provenance: Optional[str] = None,
    ) -> None:
        """
        An IndexCopy object provides a mechanism for launching explicit
        region-to-region copies between many different subregions
        simultaneously.  Note: you should NOT use these for trying to move data
        between memories!  Copy launchers should only be used for logically
        moving data between different fields.

        Parameters
        ----------
        domain : Rect
            The domain of points for the index space launch
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_index_copy_launcher_create(
            domain.raw(), legion.legion_predicate_true(), mapper, tag
        )
        if provenance is not None:
            legion.legion_index_copy_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_copy_launcher_destroy
        )
        self.src_req_index = 0
        self.dst_req_index = 0

    def set_possible_src_indirect_out_of_range(self, flag: bool) -> None:
        """
        For gather indirection copies indicate whether any of the
        source indirection pointers may point out of bounds in the
        source logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether source pointers may point outside the privileges
        """
        legion.legion_index_copy_launcher_set_possible_src_indirect_out_of_range(  # noqa: E501
            self._launcher, flag
        )

    def set_possible_dst_indirect_out_of_range(self, flag: bool) -> None:
        """
        For scatter indirection copies indicate whether any of the
        destination indirection pointers may point out of bounds in the
        destination logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether destination pointers may point outside
            the privileges
        """
        legion.legion_index_copy_launcher_set_possible_dst_indirect_out_of_range(  # noqa: E501
            self._launcher, flag
        )

    def add_src_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a source region requirement to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            source of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as sources of the copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_copy_launcher_add_src_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_ONLY,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_copy_launcher_add_src_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_ONLY,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_index_copy_launcher_add_src_field(
                self.launcher,
                self.src_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.src_req_index += 1

    def add_dst_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        redop: int = 0,
        privilege: int = legion.LEGION_WRITE_DISCARD,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a destination region requirement to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            source of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as sources of the copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        redop : int
            Optional reduction operator ID to reduce the destination fields
        privilege : int
            Optional privilege for the destination. Ignored when `redop` is
            non-zero. WRITE_DISCARD by default.
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            if redop == 0:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_region(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    privilege,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
            else:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    redop,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
        elif isinstance(upper_bound, Partition):
            if redop == 0:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_partition(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    privilege,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
            else:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    redop,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
        else:
            raise TypeError("'upper_bound' must be a Region or a Partition")
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_index_copy_launcher_add_dst_field(
                self.launcher,
                self.dst_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.dst_req_index += 1

    def add_src_indirect_requirement(
        self,
        upper_bound: Union[Region, Partition],
        field: Union[int, FieldID],
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        is_range: bool = False,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a source indirection region requirement to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            source of the index copy
        field : int or FieldID
            The ID of the field to serve as a source indirection of
            the index copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_copy_launcher_add_src_indirect_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_copy_launcher_add_src_indirect_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")

    def add_dst_indirect_requirement(
        self,
        upper_bound: Union[Region, Partition],
        field: Union[int, FieldID],
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        is_range: bool = False,
        coherence: int = legion.LEGION_EXCLUSIVE,
        **kwargs: Any,
    ) -> None:
        """
        Add a destination indirection region requirement
        to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            destination of the index copy
        field : int or FieldID
            The ID of the field to serve as a source indirection of
            the index copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")

    def set_sharding_space(self, space: IndexSpace) -> None:
        """
        Set the sharding space to use for this index copy launch
        with control replication

        Parameters
        ----------
        space : IndexSpace
            The index space to use as the sharding space
        """
        legion.legion_index_copy_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_mapper_arg(self, data: Any, size: int) -> None:
        legion.legion_index_copy_launcher_set_mapper_arg(
            self.launcher,
            (ffi.from_buffer(data), size),
        )
        # Hold a reference to the data to prevent collection
        self.data = data

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        if self.src_req_index != self.dst_req_index:
            raise RuntimeError(
                "Number of source and destination requirements "
                + "must match for copies"
            )
        legion.legion_index_copy_launcher_execute(
            runtime, context, self.launcher
        )


class Attach(Dispatchable[PhysicalRegion]):
    def __init__(
        self,
        region: Region,
        field: Union[int, FieldID],
        data: memoryview,
        mapper: int = 0,
        tag: int = 0,
        read_only: bool = False,
        provenance: Optional[str] = None,
    ) -> None:
        """
        An Attach object provides a mechanism for attaching external data to
        a logical region, thereby allowing Legion to use external data in
        place for performing computations.

        Parameters
        ----------
        region : Region
            The logical region to which external data will be attached
        field : int or FieldID
            The field ID to which the data will be attached
        data : memoryview
            Input data in a memoryview
        mapper : int
            ID of the mapper to use for mapping the operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        read_only : bool
            Whether this buffer should only be accessed read-only
        """
        self.launcher = legion.legion_attach_launcher_create(
            region.handle, region.handle, legion.LEGION_EXTERNAL_INSTANCE
        )
        if provenance is not None:
            legion.legion_attach_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self.region = region
        self._launcher = ffi.gc(
            self.launcher, legion.legion_attach_launcher_destroy
        )
        if not data.contiguous:
            raise RuntimeError("Can only attach to C- or F-contiguous buffers")
        legion.legion_attach_launcher_add_cpu_soa_field(
            self.launcher,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            ffi.from_buffer(data),
            # `not c_contiguous` implies `f_contiguous`; doing it this way so
            # that 0d/1d arrays, which are both c_ and f_contiguous, are
            # attached as C-ordered
            not data.c_contiguous,
        )

    def set_restricted(self, restricted: bool) -> None:
        """
        Set whether restricted coherence should be used on the logical region.
        If restricted coherence is enabled, changes to the data in the logical
        region will be eagerly reflected back to the external buffer.
        """
        legion.legion_attach_launcher_set_restricted(self.launcher, restricted)

    def set_mapped(self, mapped: bool) -> None:
        """
        Set whether the resulting PhysicalRegion should be considered mapped
        in the enclosing task context.
        """
        legion.legion_attach_launcher_set_mapped(self.launcher, mapped)

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> PhysicalRegion:
        """
        Dispatch the attach operation to the runtime

        Returns
        -------
        PhysicalRegion that names the attached external resource
        """
        return PhysicalRegion(
            legion.legion_attach_launcher_execute(
                runtime, context, self.launcher
            ),
            self.region,
        )


class Detach(Dispatchable[Future]):
    def __init__(self, region: PhysicalRegion, flush: bool = True) -> None:
        """
        A Detach operation will unbind an external resource from a logical
        region.  This will also allow any outstanding mutations to the logical
        region to be flushed back to the external memory allocation.

        Parameters
        ----------
        region : PhysicalRegion
            The physical region describing an external resource to be detached
        flush : bool
            Whether to flush changes to the logical region to the
            external allocation
        """
        self.physical_region = region
        # Keep a reference to the logical region to ensure that
        # it is not deleted before this detach operation can run
        self.region = region.region
        self.flush = flush

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        unordered: bool = False,
        **kwargs: Any,
    ) -> Future:
        """
        Dispatch the detach operation to the runtime

        Returns
        -------
        Future containing no data that completes when detach operation is done
        If 'unordered' is set to true then you must call legate_task_progress
        before using the returned Future
        """
        # Check to see if we're still inside the context of the task
        # If not then we just need to leak this detach because it can't be done
        if context not in _pending_unordered:
            return Future()
        if unordered:
            future = Future()
            _pending_unordered[context].append(((self, future), type(self)))
            return future
        else:
            return Future(
                legion.legion_unordered_detach_external_resource(
                    runtime,
                    context,
                    self.physical_region.handle,
                    self.flush,
                    unordered,
                )
            )


class IndexAttach(Dispatchable[ExternalResources]):
    def __init__(
        self,
        parent: Region,
        field: Union[int, FieldID],
        shard_local_data: dict[Region, Any],
        mapper: int = 0,
        tag: int = 0,
        provenance: Optional[str] = None,
    ) -> None:
        """
        A variant of Attach that allows attaching multiple pieces of external
        data as sub-regions of the same parent region. Each piece may reside
        on a different address space.

        Parameters
        ----------
        parent : Region
            The parent region to which external data will be attached as
            sub-regions
        field : int | FieldID
            The field ID to which the data will be attached
        shard_local_data : dict[Region, memoryview]
            Maps sub-regions to buffers on the shard's local address space.
            Each sub-region will be attached to the corresponding buffer.
            Each shard should pass a set of distinct subregions, and all
            sub-regions must be disjoint.
        mapper : int
            ID of the mapper to use for mapping the operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_index_attach_launcher_create(
            parent.handle,
            legion.LEGION_EXTERNAL_INSTANCE,
            True,  # restricted
        )
        if provenance is not None:
            legion.legion_index_attach_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_attach_launcher_destroy
        )
        fields = ffi.new("legion_field_id_t[1]")
        fields[0] = field.fid if isinstance(field, FieldID) else field
        # Find a local system memory
        machine = legion.legion_machine_create()
        query = legion.legion_memory_query_create(machine)
        legion.legion_memory_query_only_kind(query, legion.SYSTEM_MEM)
        legion.legion_memory_query_local_address_space(query)
        sysmem_count = legion.legion_memory_query_count(query)
        assert sysmem_count > 0
        mem = legion.legion_memory_query_first(query)
        if sysmem_count > 1:
            # TODO: We should check if the capacity of this memory is 0
            mem = legion.legion_memory_query_next(query, mem)
        legion.legion_memory_query_destroy(query)
        legion.legion_machine_destroy(machine)
        for sub_region, buf in shard_local_data.items():
            if sub_region.parent is not None:
                assert sub_region.parent.parent is parent
            if not buf.contiguous:
                raise RuntimeError(
                    "Can only attach to C- or F-contiguous buffers"
                )
            legion.legion_index_attach_launcher_attach_array_soa(
                self.launcher,
                sub_region.handle,
                ffi.from_buffer(buf),
                # `not c_contiguous` implies `f_contiguous`; doing it this way
                # so that 0d/1d arrays, which are both c_ and f_contiguous, are
                # attached as C-ordered
                not buf.c_contiguous,
                fields,
                1,  # num_fields
                mem,
            )

    def set_restricted(self, restricted: bool) -> None:
        """
        Set whether restricted coherence should be used on the logical region.
        If restricted coherence is enabled, changes to the data in the logical
        region will be eagerly reflected back to the external buffers.
        """
        legion.legion_index_attach_launcher_set_restricted(
            self.launcher, restricted
        )

    def set_deduplicate_across_shards(self, deduplicate: bool) -> None:
        """
        Set whether the runtime should check for duplicate resources
        """
        legion.legion_index_attach_launcher_set_deduplicate_across_shards(
            self.launcher, deduplicate
        )

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> ExternalResources:
        """
        Dispatch the operation to the runtime

        Returns
        -------
        An ExternalResources object that names the attached resources
        """
        return ExternalResources(
            legion.legion_attach_external_resources(
                runtime, context, self.launcher
            )
        )


class IndexDetach(Dispatchable[Future]):
    def __init__(
        self, external_resources: ExternalResources, flush: bool = True
    ) -> None:
        """
        An IndexDetach operation will unbind a collection of external resources
        that were attached together to a logical region through an IndexAttach.
        This will also allow any outstanding mutations to the logical region to
        be flushed back to the external memory allocations.

        Parameters
        ----------
        external_resources : ExternalResources
            The external resources to be detached
        flush : bool
            Whether to flush changes to the logical region to the external
            allocations
        """
        self.external_resources = external_resources
        self.flush = flush

    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> Future:
        """
        Dispatch the operation to the runtime

        Returns
        -------
        Future containing no data that completes when the operation is done
        """
        return Future(
            legion.legion_detach_external_resources(
                runtime,
                context,
                self.external_resources.handle,
                self.flush,
                False,  # unordered
            )
        )


class Acquire(Dispatchable[None]):
    def __init__(
        self,
        region: Region,
        fields: FieldListLike,
        mapper: int = 0,
        tag: int = 0,
    ) -> None:
        """
        An Acquire operation provides a mechanism for temporarily relaxing
        restricted coherence on a logical region, thereby enabling Legion
        to manage coherence with multiple copies of the data.

        Parameters
        ----------
        region : Region
            The logical region on which to relax restricted coherence
        fields : int or FieldID or List[int] or List[FieldID]
            The fields to perform the attach on
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_acquire_launcher_create(
            region.handle,
            region.handle,
            legion.legion_predicate_true(),
            mapper,
            tag,
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_acquire_launcher_destroy
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_acquire_launcher_add_field(
                self.launcher,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
            )

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        """
        Dispatch the acquire operation to the runtime
        """
        legion.legion_acquire_launcher_execute(runtime, context, self.launcher)


class Release(Dispatchable[None]):
    def __init__(
        self,
        region: Region,
        fields: FieldListLike,
        mapper: int = 0,
        tag: int = 0,
    ) -> None:
        """
        A Release operation will undo any acquire operations by putting
        restricted coherence requirements back onto a logical region.

        Parameters
        ----------
        region : Region
            The logical region on which to relax restricted coherence
        fields : int or FieldID or List[int] or List[FieldID]
            The fields to perform the attach on
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_release_launcher_create(
            region.handle,
            region.handle,
            legion.legion_predicate_true(),
            mapper,
            tag,
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_release_launcher_destroy
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_release_launcher_add_field(
                self.launcher,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
            )

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> None:
        """
        Dispatch the release operation to the runtime
        """
        legion.legion_release_launcher_execute(runtime, context, self.launcher)
