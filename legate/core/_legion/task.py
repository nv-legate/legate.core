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
from .future import Future, FutureMap
from .geometry import Point, Rect
from .partition import Partition
from .pending import _pending_deletions
from .region import Region
from .util import Dispatchable, FieldID, Mappable, dispatch

if TYPE_CHECKING:
    from . import FieldListLike, IndexSpace, OutputRegion


class Task(Dispatchable[Future], Mappable):
    def __init__(
        self,
        task_id: int,
        data: Any = None,
        size: int = 0,
        mapper: int = 0,
        tag: int = 0,
        provenance: Optional[str] = None,
    ) -> None:
        """
        A Task object provides a mechanism for launching individual sub-tasks
        from the current parent task. For scalability we encourage the use of
        IndexTasks in the common case, but there are certainly many common
        uses of individual task launches as well.

        task_id : int
            ID of the task to launch
        data : object that implements the Python buffer protocol
            The data to pass to the task as a buffer by-value
        size : int
            Number of bytes of data to pass as a by-value argument
        mapper : int
            ID of the mapper for managing the mapping of the task
        tag : int
            Tag to pass to the mapper to provide calling context
        """
        if data:
            if size <= 0:
                raise ValueError("'size' must be positive")
            self.launcher = legion.legion_task_launcher_create_from_buffer(
                task_id,
                ffi.from_buffer(data),
                size,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
            # Hold a reference to the data to prevent collection
            self._task_arg = data
        else:
            if size != 0:
                raise ValueError("'size' must be zero if there is no 'data'")
            self.launcher = legion.legion_task_launcher_create_from_buffer(
                task_id,
                ffi.NULL,
                size,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        if provenance is not None:
            legion.legion_task_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_task_launcher_destroy
        )
        self.req_index = 0
        self.outputs: list[OutputRegion] = []

    def add_no_access_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a no-access region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_NO_ACCESS,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a read-only region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
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
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_write_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a write-discard region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_WRITE_DISCARD,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_write_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a read-write region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_READ_WRITE,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_reduction_requirement(
        self,
        region: Region,
        fields: FieldListLike,
        redop: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a reduction region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        redop : int
            ReductionOp ID that will be used in the sub-task
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region_reduction(  # noqa: E501
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
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_future(self, future: Future) -> None:
        """
        Record a future as a precondition on running this task

        Parameters
        ----------
        future : Future
            The future to record as a precondition
        """
        legion.legion_task_launcher_add_future(self.launcher, future.handle)

    def add_output(self, output: OutputRegion) -> None:
        """
        Add an output region to the region requirements for this task

        Parameters
        ----------
        output : OutputRegion
            The output region that will be determined by this index launch
        """
        self.outputs.append(output)

    def add_outputs(self, outputs: list[OutputRegion]) -> None:
        """
        Add a output regions to the region requirements for this task

        Parameters
        ----------
        outputs : List[OutputRegion]
            The output regions that will be determined by this index launch
        """
        self.outputs.extend(outputs)

    def set_point(self, point: Union[Point, Any]) -> None:
        """
        Set the point to describe this task for sharding
        with control replication

        Parameters
        ----------
        point : Point or legion_domain_point_t
            The point value to associate with this task
        """
        if isinstance(point, Point):
            legion.legion_task_launcher_set_point(self.launcher, point.raw())
        else:
            legion.legion_task_launcher_set_point(self.launcher, point)

    def set_sharding_space(self, space: IndexSpace) -> None:
        """
        Set the sharding space for this individual task launch

        Parameters
        ----------
        space : IndexSpace
            The index space for use when performing sharding of this task
        """
        legion.legion_task_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_mapper_arg(self, data: Any, size: int) -> None:
        legion.legion_task_launcher_set_mapper_arg(
            self.launcher,
            (ffi.from_buffer(data), size),
        )
        # Hold a reference to the data to prevent collection
        self._mapper_arg = data

    def set_local_function(self, local: bool) -> None:
        """
        Set a flag indicating whether this task can be considered a local
        function task. Specifically that means it only has future
        preconditions with no region arguments.

        Parameters
        ----------
        local : bool
            Whether we can treat the task as a local function task
        """
        legion.legion_task_launcher_set_local_function_task(
            self.launcher, local
        )

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> Future:
        """
        Dispatch the task launch to the runtime

        Returns
        -------
        Future that will complete when the task is done and carries
        the return value of the task if any
        """
        num_outputs = len(self.outputs)
        if num_outputs == 0:
            return Future(
                legion.legion_task_launcher_execute(
                    runtime, context, self.launcher
                )
            )
        else:
            outputs = ffi.new("legion_output_requirement_t[%d]" % num_outputs)
            for i, output in enumerate(self.outputs):
                outputs[i] = output.handle
            return Future(
                legion.legion_task_launcher_execute_outputs(
                    runtime,
                    context,
                    self.launcher,
                    outputs,
                    len(self.outputs),
                )
            )


class IndexTask(Dispatchable[Union[Future, FutureMap]], Mappable):
    point_args: Union[list[Any], None]

    def __init__(
        self,
        task_id: int,
        domain: Union[Rect, IndexSpace],
        argmap: Optional[ArgumentMap] = None,
        data: Any = None,
        size: int = 0,
        mapper: int = 0,
        tag: int = 0,
        provenance: Optional[str] = None,
    ) -> None:
        """
        An IndexTask object provides a mechnanism for launching a collection
        of tasks simultaneously as described by the points in an index space.
        The point tasks inside the index task launch can still name arbitrary
        subsets of data (although if they are interfering, the projection
        functions used must be capable of describing the dependences).

        task_id : int
            ID of the task to launch
        domain : Rect or IndexSpace
            The domain description of the tasks to make (one task per point)
        argmap : ArgumentMap
            Optional argument map for passing data to point tasks
        data : object that implements the Python buffer protocol
            Buffer of byte arguments to pass to all the point tasks
        size : int
            The number of bytes in the data buffer to pass
        mapper : int
            ID of the mapper for managing the mapping of the task
        tag : int
            Tag to pass to the mapper to provide calling context
        """
        if argmap is not None:
            self.argmap = None
        else:
            self.argmap = legion.legion_argument_map_create()
            self._argmap = ffi.gc(
                self.argmap, legion.legion_argument_map_destroy
            )
            argmap = self.argmap
        if isinstance(domain, Rect):
            domain = domain.raw()
        if data:
            if size <= 0:
                raise ValueError("'size' must be positive for 'data'")
            self.launcher = legion.legion_index_launcher_create_from_buffer(
                task_id,
                domain,
                ffi.from_buffer(data),
                size,
                argmap,
                legion.legion_predicate_true(),
                False,
                mapper,
                tag,
            )
            # Hold a reference to the data to prevent collection
            self._task_arg = data
        else:
            if size != 0:
                raise ValueError("'size' must be zero if there is no 'data'")
            self.launcher = legion.legion_index_launcher_create_from_buffer(
                task_id,
                domain,
                ffi.NULL,
                size,
                argmap,
                legion.legion_predicate_true(),
                False,
                mapper,
                tag,
            )
        if provenance is not None:
            legion.legion_index_launcher_set_provenance(
                self.launcher, provenance.encode()
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_launcher_destroy
        )
        self.req_index = 0
        self.point_args = None
        self.outputs: list[OutputRegion] = []

    def add_no_access_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
        flags: int = 0,
    ) -> None:
        """
        Add a region requirement without any access privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_NO_ACCESS,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_NO_ACCESS,
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
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a region requirement read-only access privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
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
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
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
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_write_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a region requirement with write-discard privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_WRITE_DISCARD,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_WRITE_DISCARD,
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
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_write_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a region requirement with read-write privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_WRITE,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_WRITE,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_reduction_requirement(
        self,
        upper_bound: Union[Region, Partition],
        fields: FieldListLike,
        redop: int,
        projection: int,
        parent: Optional[Region] = None,
        tag: int = 0,
        flags: int = 0,
        coherence: int = legion.LEGION_EXCLUSIVE,
    ) -> None:
        """
        Add a region requirement with reduction privileges for a reduction op

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        redop : int
            ID for a reduction operator the tasks will use
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region_reduction(  # noqa: E501
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
            legion.legion_index_launcher_add_region_requirement_logical_partition_reduction(  # noqa: E501
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
            raise TypeError("'upper_bound' must be a Region or Partition")
        fields_list = fields if isinstance(fields, list) else [fields]
        for field in fields_list:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_future(self, future: Future) -> None:
        """
        Add a future precondition to all the points in the index space launch

        Parameters
        ----------
        future : Future
            A future that will be passed as a precondition to all point tasks
        """
        legion.legion_index_launcher_add_future(self.launcher, future.handle)

    def add_point_future(self, argmap: ArgumentMap) -> None:
        """
        Add an additional argument map for passing arguments to each point
        task, each of these arguments will be appended to list of futures
        for each of the point tasks.

        Parameters
        ----------
        argmap : ArgumentMap
            Per-point arguments to be added to the list of future preconditions
        """
        legion.legion_index_launcher_add_point_future(
            self.launcher, argmap.handle
        )

    def add_output(self, output: OutputRegion) -> None:
        """
        Add an output region to the region requirements for this task

        Parameters
        ----------
        output : OutputRegion
            The output region that will be determined by this index launch
        """
        self.outputs.append(output)

    def add_outputs(self, outputs: list[OutputRegion]) -> None:
        """
        Add a output regions to the region requirements for this task

        Parameters
        ----------
        outputs : List[OutputRegion]
            The output regions that will be determined by this index launch
        """
        self.outputs.extend(outputs)

    def set_point(self, point: Point, data: Any, size: int) -> None:
        """
        Set the point argument in the argument map for the index task

        Parameters
        ----------
        point : Point
            The point to set in the argument map
        data : object that implements the Python buffer protocol
            The data to be set for the point
        size : int
            The size of the data in the buffer
        """
        if data is None:
            raise ValueError("'data' must be provided to set a point")
        if size <= 0:
            raise ValueError("A positive 'size' must be specified for 'data'")
        point_arg = ffi.new("legion_task_argument_t *")
        point_arg[0].args = ffi.from_buffer(data)
        point_arg[0].arglen = size
        legion.legion_argument_map_set_point(
            self.argmap, point.raw(), point_arg, True
        )
        if not self.point_args:
            self.point_args = list()
        self.point_args.append(point_arg)

    def set_sharding_space(self, space: IndexSpace) -> None:
        """
        Set the sharding space to use for this index launch
        with control replication

        Parameters
        ----------
        space : IndexSpace
            The index space to use as the sharding space
        """
        legion.legion_index_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_mapper_arg(self, data: Any, size: int) -> None:
        legion.legion_index_launcher_set_mapper_arg(
            self.launcher,
            (ffi.from_buffer(data), size),
        )
        # Hold a reference to the data to prevent collection
        self._mapper_arg = data

    def set_concurrent(self, concurrent: bool) -> None:
        """
        Set a flag indicating whether point tasks must execute
        concurrently. Setting true to the flag directs the runtime
        to make sure the tasks are using a concurrent variant and
        also mapped to distinct processors with concurrent
        execution guarantee (i.e., no subset of the processors execute
        other tasks).

        Parameters
        ----------
        concurrent : bool
            Whether the point tasks must run concurrently
        """
        legion.legion_index_launcher_set_concurrent(self.launcher, concurrent)

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        redop: int = 0,
        **kwargs: Any,
    ) -> Union[Future, FutureMap]:
        """
        Launch this index space task to the runtime

        Parameters
        ----------
        runtime : legion_runtime_t
            Handle to the Legion runtime
        context : legion_context_t
            Handle to the enclosing parent context
        redop : int
            ID of a reduction operation to use for reducing the
            outputs of all the point tasks down to a single value

        Returns
        -------
        FutureMap if redop==0 else Future
        """
        if redop == 0:
            num_outputs = len(self.outputs)
            if num_outputs == 0:
                return FutureMap(
                    legion.legion_index_launcher_execute(
                        runtime, context, self.launcher
                    )
                )
            else:
                outputs = ffi.new(
                    "legion_output_requirement_t[%d]" % num_outputs
                )
                for i, output in enumerate(self.outputs):
                    outputs[i] = output.handle
                return FutureMap(
                    legion.legion_index_launcher_execute_outputs(
                        runtime,
                        context,
                        self.launcher,
                        outputs,
                        num_outputs,
                    )
                )

        else:
            num_outputs = len(self.outputs)
            if num_outputs == 0:
                return Future(
                    legion.legion_index_launcher_execute_deterministic_reduction(  # noqa: E501
                        runtime, context, self.launcher, redop, True
                    )
                )
            else:
                outputs = ffi.new(
                    "legion_output_requirement_t[%d]" % num_outputs
                )
                for i, output in enumerate(self.outputs):
                    outputs[i] = output.handle
                return FutureMap(
                    legion.legion_index_launcher_execute_reduction_and_outputs(
                        runtime,
                        context,
                        self.launcher,
                        redop,
                        True,
                        outputs,
                        num_outputs,
                    )
                )


class Fence(Dispatchable[Future]):
    def __init__(self, mapping: bool = False) -> None:
        """
        A Fence operation provides a mechanism for inserting either
        mapping or execution fences into the stream of tasks and
        other operations generated by a program. A mapping fence
        will prevent reordering of operations during the mapping
        process, but has no bearing on execution. An execution fence
        will ensure all tasks and operations that come perform the
        fence are ordered with respect to all the tasks and
        operations that come after it.

        Parameters
        ----------
        mapping : bool
            Whether this is a mapping fence or not
        """
        self.mapping = mapping

    @dispatch
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> Future:
        """
        Dispatch this fence to the runtime
        """
        if self.mapping:
            return Future(
                legion.legion_runtime_issue_mapping_fence(runtime, context)
            )
        else:
            return Future(
                legion.legion_runtime_issue_execution_fence(runtime, context)
            )


class ArgumentMap:
    def __init__(
        self,
        handle: Any = None,
        future_map: Optional[Union[FutureMap, Any]] = None,
    ) -> None:
        """
        An ArgumentMap is a object that allows for the passing of
        data directly to individual point tasks in IndexTask launches.

        handle : legion_argument_map_t
            The handle of which to take ownership
        future_map : legion_future_map_t
            A future map that is currently storing the data for
            this argument map
        """
        if handle is not None:
            if future_map is not None:
                raise ValueError(
                    "Cannot supply a 'future_map' with a 'handle'"
                )
            self.handle = handle
        elif future_map is None:
            self.handle = legion.legion_argument_map_create()
        else:
            self.handle = legion.legion_argument_map_from_future_map(
                future_map.handle
            )
        self.points: list[Any] = []

    def __del__(self) -> None:
        self.destroy(unordered=True)

    def destroy(self, unordered: bool) -> None:
        """
        Eagerly destroy this ArgumentMap before the garbage collector does
        It is illegal to use the ArgumentMap after this call

        Parameters
        ----------
        unordered : bool
            Whether this ArgumentMap is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_future_map_destroy(self.handle)
        self.handle = None

    def set_point(
        self, point: Point, data: Any, size: int, replace: bool = True
    ) -> None:
        """
        Set the point argument in the argument map

        Parameters
        ----------
        point : Point
            The point to set in the argument map
        data : object that implements the Python buffer protocol
            The data to be set for the point
        size : int
            The size of the data in the buffer
        replace : bool
            Whether we should replace this argument if it already exists
        """
        arg = ffi.new("legion_task_argument_t *")
        if data is not None:
            arg[0].args = ffi.from_buffer(data)
            arg[0].arglen = size
        else:
            arg[0].args = ffi.NULL
            arg[0].arglen = 0
        legion.legion_argument_map_set_point(
            self.handle, point.raw(), arg[0], replace
        )
        self.points.append(arg)

    def set_future(
        self, point: Point, future: Future, replace: bool = True
    ) -> None:
        """
        Set the point argument in the argument map using a Future

        Parameters
        ----------
        point : Point
            The point to set in the argument map
        future : Future
            The future
        replace : bool
            Whether we should replace this argument if it already exists
        """
        legion.legion_argument_map_set_future(
            self.handle, point.raw(), future.handle, replace
        )
