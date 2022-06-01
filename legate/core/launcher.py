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

from enum import IntEnum, unique
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)

from . import (
    ArgumentMap,
    BufferBuilder,
    Copy as SingleCopy,
    Future,
    FutureMap,
    IndexCopy,
    IndexTask,
    Partition as LegionPartition,
    Task as SingleTask,
    legion,
    types as ty,
)
from .utils import OrderedSet

if TYPE_CHECKING:
    from . import (
        FieldID,
        FieldSpace,
        IndexSpace,
        OutputRegion,
        Point,
        Rect,
        Region,
    )
    from ._legion.util import FieldListLike
    from .context import Context
    from .runtime import Runtime
    from .store import RegionField, Store
    from .types import DTType


LegionTask = Union[IndexTask, SingleTask, IndexCopy, SingleCopy]


@unique
class Permission(IntEnum):
    NO_ACCESS = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    REDUCTION = 4
    SOURCE_INDIRECT = 5
    TARGET_INDIRECT = 6


Serializer = Callable[[BufferBuilder, Any], None]

_SERIALIZERS: dict[Any, Serializer] = {
    bool: BufferBuilder.pack_bool,
    ty.int8: BufferBuilder.pack_8bit_int,
    ty.int16: BufferBuilder.pack_16bit_int,
    ty.int32: BufferBuilder.pack_32bit_int,
    ty.int64: BufferBuilder.pack_64bit_int,
    ty.uint8: BufferBuilder.pack_8bit_uint,
    ty.uint16: BufferBuilder.pack_16bit_uint,
    ty.uint32: BufferBuilder.pack_32bit_uint,
    ty.uint64: BufferBuilder.pack_64bit_uint,
    ty.float32: BufferBuilder.pack_32bit_float,
    ty.float64: BufferBuilder.pack_64bit_float,
    ty.string: BufferBuilder.pack_string,
}

EntryType = Tuple[Union["Broadcast", "Partition"], int, int]


def _pack(buf: BufferBuilder, value: Any, dtype: Any, is_tuple: bool) -> None:
    if dtype not in _SERIALIZERS:
        raise ValueError(f"Unsupported data type: {dtype}")
    serializer = _SERIALIZERS[dtype]

    if is_tuple:
        if dtype == ty.string:
            raise NotImplementedError(
                "Passing a tuple of strings is not yet supported"
            )
        buf.pack_32bit_uint(len(value))
        for v in value:
            serializer(buf, v)
    else:
        serializer(buf, value)


class LauncherArg(Protocol):
    def pack(self, buf: BufferBuilder) -> None:
        ...


class ScalarArg:
    def __init__(
        self,
        core_types: ty.TypeSystem,
        value: Any,
        dtype: Union[DTType, tuple[DTType]],
        untyped: bool = True,
    ) -> None:
        self._core_types = core_types
        self._value = value
        self._dtype = dtype
        self._untyped = untyped

    def pack(self, buf: BufferBuilder) -> None:
        if isinstance(self._dtype, tuple):
            if len(self._dtype) != 1:
                raise ValueError(f"Unsupported data type: {self._dtype}")
            is_tuple = True
            dtype = self._dtype[0]
        else:
            is_tuple = False
            dtype = self._dtype

        if self._untyped:
            buf.pack_bool(is_tuple)
            buf.pack_32bit_int(self._core_types[dtype].code)

        _pack(buf, self._value, dtype, is_tuple)

    def __str__(self) -> str:
        return f"ScalarArg({self._value}, {self._dtype}, {self._untyped})"


class FutureStoreArg:
    def __init__(
        self, store: Store, read_only: bool, has_storage: bool, redop: int
    ) -> None:
        self._store = store
        self._read_only = read_only
        self._has_storage = has_storage
        self._redop = redop

    def pack(self, buf: BufferBuilder) -> None:
        self._store.serialize(buf)
        buf.pack_32bit_int(self._redop)
        buf.pack_bool(self._read_only)
        buf.pack_bool(self._has_storage)
        buf.pack_32bit_int(self._store.type.size)
        _pack(buf, self._store.extents, ty.int64, True)

    def __str__(self) -> str:
        return f"FutureStoreArg({self._store})"


class RegionFieldArg:
    @overload
    def __init__(
        self,
        analyzer: RequirementAnalyzer,
        store: Store,
        dim: int,
        req: RegionReq,
        field_id: int,
        redop: int,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        analyzer: OutputAnalyzer,
        store: Store,
        dim: int,
        req: OutputReq,
        field_id: int,
        redop: int,
    ) -> None:
        ...

    def __init__(
        self,
        analyzer: Union[OutputAnalyzer, RequirementAnalyzer],
        store: Store,
        dim: int,
        req: Union[OutputReq, RegionReq],
        field_id: int,
        redop: int,
    ) -> None:
        self._analyzer = analyzer
        self._store = store
        self._dim = dim
        self._req = req
        self._field_id = field_id
        self._redop = redop

    def pack(self, buf: BufferBuilder) -> None:
        self._store.serialize(buf)
        buf.pack_32bit_int(self._redop)
        buf.pack_32bit_int(self._dim)

        buf.pack_32bit_uint(
            self._analyzer.get_requirement_index(
                self._req, self._field_id  # type: ignore [arg-type]
            )
        )
        buf.pack_32bit_uint(self._field_id)

    def __str__(self) -> str:
        return f"RegionFieldArg({self._dim}, {self._req}, {self._field_id})"


LegionTaskMethod = Any


def _index_copy_add_rw_dst_requirement(*args: Any, **kwargs: Any) -> None:
    kwargs["privilege"] = legion.LEGION_READ_WRITE
    IndexCopy.add_dst_requirement(*args, **kwargs)


def _single_copy_add_rw_dst_requirement(*args: Any, **kwargs: Any) -> None:
    kwargs["privilege"] = legion.LEGION_READ_WRITE
    SingleCopy.add_dst_requirement(*args, **kwargs)


_single_task_calls: dict[Permission, LegionTaskMethod] = {
    Permission.NO_ACCESS: SingleTask.add_no_access_requirement,
    Permission.READ: SingleTask.add_read_requirement,
    Permission.WRITE: SingleTask.add_write_requirement,
    Permission.READ_WRITE: SingleTask.add_read_write_requirement,
    Permission.REDUCTION: SingleTask.add_reduction_requirement,
}

_index_task_calls: dict[Permission, LegionTaskMethod] = {
    Permission.NO_ACCESS: IndexTask.add_no_access_requirement,
    Permission.READ: IndexTask.add_read_requirement,
    Permission.WRITE: IndexTask.add_write_requirement,
    Permission.READ_WRITE: IndexTask.add_read_write_requirement,
    Permission.REDUCTION: IndexTask.add_reduction_requirement,
}

_index_copy_calls: dict[Permission, LegionTaskMethod] = {
    Permission.READ: IndexCopy.add_src_requirement,
    Permission.WRITE: IndexCopy.add_dst_requirement,
    Permission.READ_WRITE: _index_copy_add_rw_dst_requirement,
    Permission.SOURCE_INDIRECT: IndexCopy.add_src_indirect_requirement,
    Permission.TARGET_INDIRECT: IndexCopy.add_dst_indirect_requirement,
}

_single_copy_calls: dict[Permission, LegionTaskMethod] = {
    Permission.READ: SingleCopy.add_src_requirement,
    Permission.WRITE: SingleCopy.add_dst_requirement,
    Permission.READ_WRITE: _single_copy_add_rw_dst_requirement,
    Permission.SOURCE_INDIRECT: SingleCopy.add_src_indirect_requirement,
    Permission.TARGET_INDIRECT: SingleCopy.add_dst_indirect_requirement,
}


class Broadcast:
    # Use the same signature as Partition's constructor
    # so that the caller can construct projection objects in a uniform way
    def __init__(self, part: Optional[LegionPartition], proj: int) -> None:
        assert part is None
        self.part = part
        self.proj = proj
        self.redop: Union[int, None] = None

    def add(
        self,
        task: LegionTask,
        req: RegionReq,
        fields: FieldListLike,
        methods: dict[Permission, LegionTaskMethod],
    ) -> None:
        f = methods[req.permission]
        parent = req.region
        while parent.parent is not None:
            parent_partition = parent.parent
            parent = parent_partition.parent
        if req.permission != Permission.REDUCTION:
            f(task, req.region, fields, 0, parent=parent, tag=req.tag)
        else:
            f(
                task,
                req.region,
                fields,
                self.redop,
                0,
                parent=parent,
                tag=req.tag,
            )

    def add_single(
        self,
        task: LegionTask,
        req: RegionReq,
        fields: FieldListLike,
        methods: dict[Permission, LegionTaskMethod],
    ) -> None:
        f = methods[req.permission]
        if req.permission != Permission.REDUCTION:
            f(task, req.region, fields, tag=req.tag, flags=req.flags)
        else:
            f(
                task,
                req.region,
                fields,
                self.redop,
                tag=req.tag,
                flags=req.flags,
            )

    def __hash__(self) -> int:
        return hash(("Broadcast", self.redop))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Broadcast):
            return NotImplemented
        return self.redop == other.redop


class Partition:
    def __init__(self, part: Optional[LegionPartition], proj: int) -> None:
        assert part is not None
        self.part = part
        self.proj = proj
        self.redop = None

    def add(
        self,
        task: LegionTask,
        req: RegionReq,
        fields: FieldListLike,
        methods: dict[Permission, LegionTaskMethod],
    ) -> None:
        f = methods[req.permission]
        if req.permission != Permission.REDUCTION:
            f(task, self.part, fields, self.proj, tag=req.tag, flags=req.flags)
        else:
            f(
                task,
                self.part,
                fields,
                self.redop,
                self.proj,
                tag=req.tag,
                flags=req.flags,
            )

    def add_single(
        self,
        task: LegionTask,
        req: RegionReq,
        fields: FieldListLike,
        methods: dict[Permission, LegionTaskMethod],
    ) -> None:
        f = methods[req.permission]
        if req.permission != Permission.REDUCTION:
            f(task, req.region, fields, tag=req.tag)
        else:
            f(task, req.region, fields, self.redop, tag=req.tag)

    def __hash__(self) -> int:
        return hash((self.part, self.proj, self.redop))

    def __repr__(self) -> str:
        return repr((self.part, self.proj, self.redop))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return (
            self.part == other.part
            and self.proj == other.proj
            and self.redop == other.redop
        )


Proj = Union[Broadcast, Partition]  # TODO: (bev) Protocol


class RegionReq:
    def __init__(
        self,
        region: Region,
        permission: Permission,
        proj: Proj,
        tag: int,
        flags: int,
    ) -> None:
        self.region = region
        self.permission = permission
        self.proj = proj
        self.tag = tag
        self.flags = flags

    def __str__(self) -> str:
        return (
            f"RegionReq({self.region}, {self.permission}, "
            f"{self.proj}, {self.tag}, {self.flags})"
        )

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(
            (self.region, self.permission, self.proj, self.tag, self.flags)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegionReq):
            return NotImplemented
        return (
            self.region == other.region
            and self.proj == other.proj
            and self.permission == other.permission
            and self.tag == other.tag
            and self.flags == other.flags
        )

    def promote_to_read_write(self) -> RegionReq:
        return RegionReq(
            self.region, Permission.READ_WRITE, self.proj, self.tag, self.flags
        )


class OutputReq:
    def __init__(
        self, runtime: Runtime, fspace: FieldSpace, ndim: int
    ) -> None:
        self.runtime = runtime
        self.fspace = fspace
        self.ndim = ndim
        self.output_region: Union[OutputRegion, None] = None

    def __str__(self) -> str:
        return f"OutputReq({self.fspace}, {self.ndim})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.fspace, self.ndim))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputReq):
            return NotImplemented
        return self.fspace == other.fspace and self.ndim == other.ndim

    def _create_output_region(self, fields: FieldListLike) -> None:
        assert self.output_region is None
        self.output_region = self.runtime.create_output_region(
            self.fspace, fields, self.ndim
        )

    def add(self, task: Any, fields: FieldListLike) -> None:
        self._create_output_region(fields)
        task.add_output(self.output_region)

    def add_single(self, task: Any, fields: FieldListLike) -> None:
        self._create_output_region(fields)
        task.add_output(self.output_region)

    def update_storage(self, store: Store, field_id: int) -> None:
        assert self.output_region is not None
        region_field = self.runtime.import_output_region(
            self.output_region,
            field_id,
            store.type,
        )
        store.set_storage(region_field)


class ProjectionSet:
    def __init__(self) -> None:
        self._entries: dict[Permission, OrderedSet[EntryType]] = {}

    def _create(self, perm: Permission, entry: EntryType) -> None:
        self._entries[perm] = OrderedSet([entry])

    def _update(self, perm: Permission, entry: EntryType) -> None:
        entries = self._entries[perm]
        entries.add(entry)
        if perm == Permission.WRITE and len(entries) > 1:
            raise ValueError("Interfering requirements found")

    def insert(self, perm: Permission, proj_info: EntryType) -> None:
        if perm == Permission.READ_WRITE:
            self.insert(Permission.READ, proj_info)
            self.insert(Permission.WRITE, proj_info)
        else:
            if perm in self._entries:
                self._update(perm, proj_info)
            else:
                self._create(perm, proj_info)

    def coalesce(
        self, error_on_interference: bool
    ) -> Union[
        list[tuple[Permission, Union[Broadcast, Partition], int, int]],
        list[tuple[Permission, OrderedSet[EntryType]]],
    ]:
        if len(self._entries) == 1:
            perm = list(self._entries.keys())[0]
            return [(perm, *entry) for entry in self._entries[perm]]
        all_perms = OrderedSet(self._entries.keys())
        # If the fields is requested with conflicting permissions,
        # promote them to read write permission.
        if len(all_perms - OrderedSet([Permission.NO_ACCESS])) > 1:
            perm = Permission.READ_WRITE

            # When the field requires read write permission,
            # all projections must be the same
            all_entries: OrderedSet[EntryType] = OrderedSet()
            for entry in self._entries.values():
                all_entries.update(entry)
            if len(all_entries) > 1:
                if error_on_interference:
                    raise ValueError(
                        f"Interfering requirements found: {all_entries}"
                    )
                else:
                    results: list[
                        tuple[
                            Permission, Union[Broadcast, Partition], int, int
                        ]
                    ] = []
                    for all_entry in all_entries:
                        results.append((perm, *all_entry))
                    return results

            return [(perm, *all_entries.pop())]

        # This can happen when there is a no access requirement.
        # For now, we don't coalesce it with others.
        else:
            return [pair for pair in self._entries.items()]

    def __repr__(self) -> str:
        return str(self._entries)


class FieldSet:
    def __init__(self) -> None:
        self._fields: dict[int, ProjectionSet] = {}

    def insert(
        self, field_id: int, perm: Permission, proj_info: EntryType
    ) -> None:
        if field_id in self._fields:
            proj_set = self._fields[field_id]
        else:
            proj_set = ProjectionSet()
            self._fields[field_id] = proj_set
        proj_set.insert(perm, proj_info)

    def coalesce(
        self, error_on_interference: bool
    ) -> dict[Any, list[Union[int, FieldID]]]:
        coalesced: dict[Any, list[Union[int, FieldID]]] = {}
        for field_id, proj_set in self._fields.items():
            proj_infos = proj_set.coalesce(error_on_interference)
            for key in proj_infos:
                if key in coalesced:
                    coalesced[key].append(field_id)
                else:
                    coalesced[key] = [field_id]

        return coalesced


class RequirementAnalyzer:
    def __init__(self, error_on_interference: bool = True) -> None:
        self._field_sets: dict[Any, FieldSet] = {}
        self._requirements: list[tuple[RegionReq, Any]] = []
        self._requirement_map: dict[
            tuple[RegionReq, Union[int, FieldID]], int
        ] = {}
        self._error_on_interference = error_on_interference

    @property
    def requirements(self) -> list[tuple[RegionReq, Any]]:
        return self._requirements

    @property
    def empty(self) -> bool:
        return len(self._field_sets) == 0

    def __del__(self) -> None:
        self._field_sets.clear()
        self._requirements.clear()
        self._requirement_map.clear()

    def insert(self, req: RegionReq, field_id: int) -> None:
        region = req.region
        field_set = self._field_sets.get(region)
        if field_set is None:
            field_set = FieldSet()
            self._field_sets[region] = field_set
        proj_info = (req.proj, req.tag, req.flags)
        field_set.insert(field_id, req.permission, proj_info)

    def analyze_requirements(self) -> None:
        for region, field_set in self._field_sets.items():
            perm_map = field_set.coalesce(self._error_on_interference)
            for key, fields in perm_map.items():
                req_idx = len(self._requirements)
                req = RegionReq(region, *key)
                for field_id in fields:
                    self._requirement_map[(req, field_id)] = req_idx
                self._requirements.append((req, fields))

    def get_requirement_index(self, req: RegionReq, field_id: int) -> int:
        try:
            return self._requirement_map[(req, field_id)]
        except KeyError:
            req = req.promote_to_read_write()
            return self._requirement_map[(req, field_id)]


class OutputAnalyzer:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._groups: dict[Any, OrderedSet[tuple[int, Store]]] = {}
        self._requirements: list[tuple[OutputReq, Any]] = []
        self._requirement_map: dict[tuple[OutputReq, int], int] = {}

    @property
    def requirements(self) -> list[tuple[OutputReq, Any]]:
        return self._requirements

    @property
    def empty(self) -> bool:
        return len(self._groups) == 0

    def __del__(self) -> None:
        self._groups.clear()
        self._requirements.clear()
        self._requirement_map.clear()

    def insert(self, req: OutputReq, field_id: int, store: Store) -> None:
        group = self._groups.get(req)
        if group is None:
            group = OrderedSet()
            self._groups[req] = group
        group.add((field_id, store))

    def analyze_requirements(self) -> None:
        for req, group in self._groups.items():
            req_idx = len(self._requirements)
            fields = []
            field_set: OrderedSet[int] = OrderedSet()
            for field_id, store in group:
                self._requirement_map[(req, field_id)] = req_idx
                if field_id in field_set:
                    raise RuntimeError(
                        f"{field_id} is duplicated in output requirement {req}"
                    )
                fields.append(field_id)
                field_set.add(field_id)

            self._requirements.append((req, fields))

    def get_requirement_index(self, req: OutputReq, field_id: int) -> int:
        return self._requirement_map[(req, field_id)]

    def update_storages(self) -> None:
        for req, group in self._groups.items():
            for field_id, store in group:
                req.update_storage(store, field_id)


class TaskLauncher:
    def __init__(
        self,
        context: Context,
        task_id: int,
        mapper_id: int = 0,
        tag: int = 0,
        error_on_interference: bool = True,
        side_effect: bool = False,
    ) -> None:
        assert type(tag) != bool
        self._context = context
        self._runtime = context.runtime
        self._core_types = self._runtime.core_context.type_system
        self._task_id = task_id
        self._mapper_id = mapper_id
        self._inputs: list[LauncherArg] = []
        self._outputs: list[LauncherArg] = []
        self._reductions: list[LauncherArg] = []
        self._scalars: list[ScalarArg] = []
        self._comms: list[FutureMap] = []
        self._req_analyzer = RequirementAnalyzer(error_on_interference)
        self._out_analyzer = OutputAnalyzer(context.runtime)
        self._future_args: list[Future] = []
        self._future_map_args: list[FutureMap] = []
        self._tag = tag
        self._sharding_space: Union[IndexSpace, None] = None
        self._point: Union[Point, None] = None
        self._output_regions: list[OutputRegion] = []
        self._error_on_interference = error_on_interference
        self._has_side_effect = side_effect
        self._insert_barrier = False

    @property
    def library_task_id(self) -> int:
        return self._task_id

    @property
    def library_mapper_id(self) -> int:
        return self._mapper_id

    @property
    def legion_task_id(self) -> int:
        return self._context.get_task_id(self._task_id)

    @property
    def legion_mapper_id(self) -> int:
        return self._context.get_mapper_id(self._mapper_id)

    def __del__(self) -> None:
        del self._req_analyzer
        del self._out_analyzer
        self._future_args.clear()
        self._future_map_args.clear()
        self._output_regions.clear()

    def add_scalar_arg(
        self,
        value: Any,
        dtype: DTType,
        untyped: bool = True,
    ) -> None:
        self._scalars.append(
            ScalarArg(self._core_types, value, dtype, untyped)
        )

    def add_store(
        self,
        args: list[LauncherArg],
        store: Store,
        proj: Proj,
        perm: Permission,
        tag: int,
        flags: int,
    ) -> None:
        redop = -1 if proj.redop is None else proj.redop
        if store.kind is Future:
            if TYPE_CHECKING:
                assert isinstance(store.storage, Future)

            has_storage = perm != Permission.WRITE
            read_only = perm == Permission.READ
            if has_storage:
                self.add_future(store.storage)
            args.append(FutureStoreArg(store, read_only, has_storage, redop))

        else:
            if TYPE_CHECKING:
                assert isinstance(store.storage, RegionField)

            region = store.storage.region
            field_id = store.storage.field.field_id

            req = RegionReq(region, perm, proj, tag, flags)

            self._req_analyzer.insert(req, field_id)
            args.append(
                RegionFieldArg(
                    self._req_analyzer,
                    store,
                    region.index_space.get_dim(),
                    req,
                    field_id,
                    redop,
                )
            )

    def add_input(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(self._inputs, store, proj, Permission.READ, tag, flags)

    def add_output(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(
            self._outputs, store, proj, Permission.WRITE, tag, flags
        )

    def add_reduction(
        self,
        store: Store,
        proj: Proj,
        tag: int = 0,
        flags: int = 0,
        read_write: bool = False,
    ) -> None:
        if read_write and store.kind is not Future:
            self.add_store(
                self._reductions,
                store,
                proj,
                Permission.READ_WRITE,
                tag,
                flags,
            )
        else:
            self.add_store(
                self._reductions, store, proj, Permission.REDUCTION, tag, flags
            )

    def add_unbound_output(
        self, store: Store, fspace: FieldSpace, field_id: int
    ) -> None:
        req = OutputReq(self._runtime, fspace, store.ndim)

        self._out_analyzer.insert(req, field_id, store)

        self._outputs.append(
            RegionFieldArg(
                self._out_analyzer,
                store,
                store.ndim,
                req,
                field_id,
                -1,
            )
        )

    def add_future(self, future: Future) -> None:
        self._future_args.append(future)

    def add_future_map(self, future_map: FutureMap) -> None:
        self._future_map_args.append(future_map)

    def add_communicator(self, handle: FutureMap) -> None:
        self._comms.append(handle)

    def insert_barrier(self) -> None:
        self._insert_barrier = True

    def set_sharding_space(self, space: IndexSpace) -> None:
        self._sharding_space = space

    def set_point(self, point: Point) -> None:
        self._point = point

    @staticmethod
    def pack_args(
        argbuf: BufferBuilder,
        args: Sequence[LauncherArg],
    ) -> None:
        argbuf.pack_32bit_uint(len(args))
        for arg in args:
            arg.pack(argbuf)

    def build_task(
        self, launch_domain: Rect, argbuf: BufferBuilder
    ) -> IndexTask:
        self._req_analyzer.analyze_requirements()
        self._out_analyzer.analyze_requirements()

        self.pack_args(argbuf, self._inputs)
        self.pack_args(argbuf, self._outputs)
        self.pack_args(argbuf, self._reductions)
        self.pack_args(argbuf, self._scalars)
        argbuf.pack_bool(self._insert_barrier)
        argbuf.pack_32bit_uint(len(self._comms))

        task = IndexTask(
            self.legion_task_id,
            launch_domain,
            self._context.empty_argmap,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self.legion_mapper_id,
            tag=self._tag,
        )
        if self._sharding_space is not None:
            task.set_sharding_space(self._sharding_space)

        for (req, fields) in self._req_analyzer.requirements:
            req.proj.add(task, req, fields, _index_task_calls)
        for future in self._future_args:
            task.add_future(future)
        if self._insert_barrier:
            volume = launch_domain.get_volume()
            arrival, wait = self._runtime.get_barriers(volume)
            task.add_future(arrival)
            task.add_future(wait)
        for (out_req, fields) in self._out_analyzer.requirements:
            out_req.add(task, fields)
        for comm in self._comms:
            task.add_point_future(ArgumentMap(future_map=comm))
        for future_map in self._future_map_args:
            task.add_point_future(ArgumentMap(future_map=future_map))
        return task

    def build_single_task(self, argbuf: BufferBuilder) -> SingleTask:
        self._req_analyzer.analyze_requirements()
        self._out_analyzer.analyze_requirements()

        self.pack_args(argbuf, self._inputs)
        self.pack_args(argbuf, self._outputs)
        self.pack_args(argbuf, self._reductions)
        self.pack_args(argbuf, self._scalars)

        assert len(self._comms) == 0

        task = SingleTask(
            self.legion_task_id,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self.legion_mapper_id,
            tag=self._tag,
        )
        for (req, fields) in self._req_analyzer.requirements:
            req.proj.add_single(task, req, fields, _single_task_calls)
        for future in self._future_args:
            task.add_future(future)
        for (out_req, fields) in self._out_analyzer.requirements:
            out_req.add_single(task, fields)
        if (
            not self._has_side_effect
            and self._req_analyzer.empty
            and self._out_analyzer.empty
        ):
            task.set_local_function(True)
        if self._sharding_space is not None:
            task.set_sharding_space(self._sharding_space)
        if self._point is not None:
            task.set_point(self._point)
        return task

    def execute(self, launch_domain: Rect) -> FutureMap:
        task = self.build_task(launch_domain, BufferBuilder())
        result = self._context.dispatch(task)
        assert isinstance(result, FutureMap)
        self._out_analyzer.update_storages()
        return result

    def execute_single(self) -> Future:
        argbuf = BufferBuilder()
        result = self._context.dispatch_single(self.build_single_task(argbuf))
        self._out_analyzer.update_storages()
        return result


class CopyLauncher:
    def __init__(
        self,
        context: Context,
        source_oor: bool = True,
        target_oor: bool = True,
        mapper_id: int = 0,
        tag: int = 0,
    ) -> None:
        assert type(tag) != bool
        self._context = context
        self._runtime = context.runtime
        self._mapper_id = mapper_id
        self._req_analyzer = RequirementAnalyzer()
        self._tag = tag
        self._sharding_space: Union[IndexSpace, None] = None
        self._point: Union[Point, None] = None
        self._source_oor = source_oor
        self._target_oor = target_oor

    @property
    def library_mapper_id(self) -> int:
        return self._mapper_id

    @property
    def legion_mapper_id(self) -> int:
        return self._context.get_mapper_id(self._mapper_id)

    def __del__(self) -> None:
        del self._req_analyzer

    def add_store(
        self, store: Store, proj: Proj, perm: Permission, tag: int, flags: int
    ) -> None:
        assert store.kind is not Future
        assert store._transform.bottom

        if TYPE_CHECKING:
            assert isinstance(store.storage, RegionField)

        region = store.storage.region
        field_id = store.storage.field.field_id

        req = RegionReq(region, perm, proj, tag, flags)

        self._req_analyzer.insert(req, field_id)

    def add_input(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(store, proj, Permission.READ, tag, flags)

    def add_output(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(store, proj, Permission.WRITE, tag, flags)

    def add_inout(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(store, proj, Permission.READ_WRITE, tag, flags)

    def add_reduction(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(store, proj, Permission.REDUCTION, tag, flags)

    def add_source_indirect(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(store, proj, Permission.SOURCE_INDIRECT, tag, flags)

    def add_target_indirect(
        self, store: Store, proj: Proj, tag: int = 0, flags: int = 0
    ) -> None:
        self.add_store(store, proj, Permission.TARGET_INDIRECT, tag, flags)

    def set_sharding_space(self, space: IndexSpace) -> None:
        self._sharding_space = space

    def set_point(self, point: Point) -> None:
        self._point = point

    def build_copy(self, launch_domain: Rect) -> IndexCopy:
        self._req_analyzer.analyze_requirements()

        copy = IndexCopy(
            launch_domain,
            mapper=self.legion_mapper_id,
            tag=self._tag,
        )
        for (req, fields) in self._req_analyzer.requirements:
            if req.permission in (
                Permission.SOURCE_INDIRECT,
                Permission.TARGET_INDIRECT,
            ):
                assert len(fields) == 1
                fields = fields[0]
            req.proj.add(copy, req, fields, _index_copy_calls)
        if self._sharding_space is not None:
            copy.set_sharding_space(self._sharding_space)
        copy.set_possible_src_indirect_out_of_range(self._source_oor)
        copy.set_possible_dst_indirect_out_of_range(self._target_oor)
        return copy

    def build_single_copy(self) -> SingleCopy:
        self._req_analyzer.analyze_requirements()

        copy = SingleCopy(
            mapper=self.legion_mapper_id,
            tag=self._tag,
        )
        for (req, fields) in self._req_analyzer.requirements:
            if req.permission in (
                Permission.SOURCE_INDIRECT,
                Permission.TARGET_INDIRECT,
            ):
                assert len(fields) == 1
                fields = fields[0]
            req.proj.add_single(copy, req, fields, _single_copy_calls)
        if self._sharding_space is not None:
            copy.set_sharding_space(self._sharding_space)
        if self._point is not None:
            copy.set_point(self._point)
        copy.set_possible_src_indirect_out_of_range(self._source_oor)
        copy.set_possible_dst_indirect_out_of_range(self._target_oor)
        return copy

    def execute(
        self, launch_domain: Rect, redop: Optional[int] = None
    ) -> None:
        copy = self.build_copy(launch_domain)
        self._context.dispatch(copy)

    def execute_single(self) -> None:
        copy = self.build_single_copy()
        self._context.dispatch_single(copy)
