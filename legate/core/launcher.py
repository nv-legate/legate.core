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

import legate.core.types as ty

from .legion import (
    ArgumentMap,
    BufferBuilder,
    Copy as SingleCopy,
    Future,
    IndexCopy,
    IndexTask,
    Task as SingleTask,
)


@unique
class Permission(IntEnum):
    NO_ACCESS = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    REDUCTION = 4
    SOURCE_INDIRECT = 5
    TARGET_INDIRECT = 6


_SERIALIZERS = {
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
}


def _pack(buf, value, dtype, is_tuple):
    if dtype not in _SERIALIZERS:
        raise ValueError(f"Unsupported data type: {dtype}")
    serializer = _SERIALIZERS[dtype]

    if is_tuple:
        buf.pack_32bit_uint(len(value))
        for v in value:
            serializer(buf, v)
    else:
        serializer(buf, value)


class ScalarArg(object):
    def __init__(self, core_types, value, dtype, untyped=True):
        self._core_types = core_types
        self._value = value
        self._dtype = dtype
        self._untyped = untyped

    def pack(self, buf):
        if isinstance(self._dtype, tuple):
            if len(self._dtype) != 1:
                raise ValueError(
                    "Unsupported data type: %s" % str(self._dtype)
                )
            is_tuple = True
            dtype = self._dtype[0]
        else:
            is_tuple = False
            dtype = self._dtype

        if self._untyped:
            buf.pack_bool(is_tuple)
            buf.pack_32bit_int(self._core_types[dtype].code)

        _pack(buf, self._value, dtype, is_tuple)

    def __str__(self):
        return f"ScalarArg({self._value}, {self._dtype}, {self._untyped})"

    def __repr__(self):
        return str(self)


class FutureStoreArg(object):
    def __init__(self, store, read_only, has_storage):
        self._store = store
        self._read_only = read_only
        self._has_storage = has_storage

    def pack(self, buf):
        self._store.serialize(buf)
        buf.pack_bool(self._read_only)
        buf.pack_bool(self._has_storage)
        buf.pack_32bit_int(self._store.type.size)
        _pack(buf, self._store.extents, ty.int64, True)

    def __str__(self):
        return f"FutureStoreArg({self._store})"

    def __repr__(self):
        return str(self)


class RegionFieldArg(object):
    def __init__(self, analyzer, store, dim, req, field_id, redop):
        self._analyzer = analyzer
        self._store = store
        self._dim = dim
        self._req = req
        self._field_id = field_id
        self._redop = redop

    def pack(self, buf):
        self._store.serialize(buf)
        buf.pack_32bit_int(self._redop)
        buf.pack_32bit_int(self._dim)
        buf.pack_32bit_uint(
            self._analyzer.get_requirement_index(self._req, self._field_id)
        )
        buf.pack_32bit_uint(self._field_id)

    def __str__(self):
        return f"RegionFieldArg({self._dim}, {self._req}, {self._field_id})"

    def __repr__(self):
        return str(self)


_single_task_calls = {
    Permission.NO_ACCESS: SingleTask.add_no_access_requirement,
    Permission.READ: SingleTask.add_read_requirement,
    Permission.WRITE: SingleTask.add_write_requirement,
    Permission.READ_WRITE: SingleTask.add_read_write_requirement,
    Permission.REDUCTION: SingleTask.add_reduction_requirement,
}

_index_task_calls = {
    Permission.NO_ACCESS: IndexTask.add_no_access_requirement,
    Permission.READ: IndexTask.add_read_requirement,
    Permission.WRITE: IndexTask.add_write_requirement,
    Permission.READ_WRITE: IndexTask.add_read_write_requirement,
    Permission.REDUCTION: IndexTask.add_reduction_requirement,
}

_index_copy_calls = {
    Permission.READ: IndexCopy.add_src_requirement,
    Permission.WRITE: IndexCopy.add_dst_requirement,
    Permission.SOURCE_INDIRECT: IndexCopy.add_src_indirect_requirement,
    Permission.TARGET_INDIRECT: IndexCopy.add_dst_indirect_requirement,
}

_single_copy_calls = {
    Permission.READ: SingleCopy.add_src_requirement,
    Permission.WRITE: SingleCopy.add_dst_requirement,
    Permission.SOURCE_INDIRECT: SingleCopy.add_src_indirect_requirement,
    Permission.TARGET_INDIRECT: SingleCopy.add_dst_indirect_requirement,
}


class Broadcast(object):
    __slots__ = ["part", "proj", "redop"]

    # Use the same signature as Partition's constructor
    # so that the caller can construct projection objects in a uniform way
    def __init__(self, part, proj):
        self.part = part
        self.proj = proj
        self.redop = None

    def add(self, task, req, fields, methods):
        f = methods[req.permission]
        parent = req.region
        while parent.parent is not None:
            parent = parent.parent
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

    def add_single(self, task, req, fields, methods):
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

    def __hash__(self):
        return hash(("Broadcast", self.redop))

    def __eq__(self, other):
        return isinstance(other, Broadcast) and self.redop == other.redop


class Partition(object):
    __slots__ = ["part", "proj", "redop"]

    def __init__(self, part, proj):
        self.part = part
        self.proj = proj
        self.redop = None

    def add(self, task, req, fields, methods):
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

    def add_single(self, task, req, fields, methods):
        f = methods[req.permission]
        if req.permission != Permission.REDUCTION:
            f(task, req.region, fields, tag=req.tag)
        else:
            f(task, req.region, fields, self.redop, tag=req.tag)

    def __hash__(self):
        return hash((self.part, self.proj, self.redop))

    def __repr__(self):
        return repr((self.part, self.proj, self.redop))

    def __eq__(self, other):
        return (
            isinstance(other, Partition)
            and self.part == other.part
            and self.proj == other.proj
            and self.redop == other.redop
        )


class RegionReq(object):
    def __init__(self, region, permission, proj, tag, flags):
        self.region = region
        self.permission = permission
        self.proj = proj
        self.tag = tag
        self.flags = flags

    def __str__(self):
        return (
            f"RegionReq({self.region}, {self.permission}, "
            f"{self.proj}, {self.tag}, {self.flags})"
        )

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(
            (self.region, self.permission, self.proj, self.tag, self.flags)
        )

    def __eq__(self, other):
        return (
            self.region == other.region
            and self.proj == other.proj
            and self.permission == other.permission
            and self.tag == other.tag
            and self.flags == other.flags
        )

    def promote_to_read_write(self):
        return RegionReq(
            self.region, Permission.READ_WRITE, self.proj, self.tag, self.flags
        )


class OutputReq(object):
    def __init__(self, runtime, fspace):
        self.runtime = runtime
        self.fspace = fspace
        self.output_region = None

    def __str__(self):
        return f"OutputReq({self.fspace})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.fspace)

    def __eq__(self, other):
        return self.fspace == other.fspace

    def _create_output_region(self, fields):
        assert self.output_region is None
        self.output_region = self.runtime.create_output_region(
            self.fspace, fields
        )

    def add(self, task, fields):
        self._create_output_region(fields)
        task.add_output(self.output_region)

    def add_single(self, task, fields):
        self._create_output_region(fields)
        task.add_output(self.output_region)

    def update_storage(self, store, field_id):
        assert self.output_region is not None
        region_field = self.runtime.import_output_region(
            self.output_region,
            field_id,
            store.type,
        )
        store.set_storage(region_field)


class ProjectionSet(object):
    def __init__(self):
        self._entries = {}

    def _create(self, perm, entry):
        self._entries[perm] = set([entry])

    def _update(self, perm, entry):
        entries = self._entries[perm]
        entries.add(entry)
        if perm == Permission.WRITE and len(entries) > 1:
            raise ValueError("Interfering requirements found")

    def insert(self, perm, proj_info):
        if perm == Permission.READ_WRITE:
            self.insert(Permission.READ, proj_info)
            self.insert(Permission.WRITE, proj_info)
        else:
            if perm in self._entries:
                self._update(perm, proj_info)
            else:
                self._create(perm, proj_info)

    def coalesce(self, error_on_interference):
        if len(self._entries) == 1:
            perm = list(self._entries.keys())[0]
            return [(perm, *entry) for entry in self._entries[perm]]
        all_perms = set(self._entries.keys())
        # If the fields is requested with conflicting permissions,
        # promote them to read write permission.
        if len(all_perms - set([Permission.NO_ACCESS])) > 1:
            perm = Permission.READ_WRITE

            # When the field requires read write permission,
            # all projections must be the same
            all_entries = set()
            for entry in self._entries.values():
                all_entries = all_entries | entry
            if len(all_entries) > 1:
                if error_on_interference:
                    raise ValueError(
                        f"Interfering requirements found: {all_entries}"
                    )
                else:
                    results = []
                    for entry in all_entries:
                        results.append((perm, *entry))
                    return results

            return [(perm, *all_entries.pop())]

        # This can happen when there is a no access requirement.
        # For now, we don't coalesce it with others.
        else:
            return [pair for pair in self._entries.items()]

    def __repr__(self):
        return str(self._entries)


class FieldSet(object):
    def __init__(self):
        self._fields = {}

    def insert(self, field_id, perm, proj_info):
        if field_id in self._fields:
            proj_set = self._fields[field_id]
        else:
            proj_set = ProjectionSet()
            self._fields[field_id] = proj_set
        proj_set.insert(perm, proj_info)

    def coalesce(self, error_on_interference):
        coalesced = {}
        for field_id, proj_set in self._fields.items():
            proj_infos = proj_set.coalesce(error_on_interference)
            for key in proj_infos:
                if key in coalesced:
                    coalesced[key].append(field_id)
                else:
                    coalesced[key] = [field_id]

        return coalesced


class RequirementAnalyzer(object):
    def __init__(self, error_on_interference=True):
        self._field_sets = {}
        self._requirements = []
        self._requirement_map = {}
        self._error_on_interference = error_on_interference

    @property
    def requirements(self):
        return self._requirements

    @property
    def empty(self):
        return len(self._field_sets) == 0

    def __del__(self):
        self._field_sets.clear()
        self._requirements.clear()
        self._requirement_map.clear()

    def insert(self, req, field_id):
        region = req.region
        field_set = self._field_sets.get(region)
        if field_set is None:
            field_set = FieldSet()
            self._field_sets[region] = field_set
        proj_info = (req.proj, req.tag, req.flags)
        field_set.insert(field_id, req.permission, proj_info)

    def analyze_requirements(self):
        for region, field_set in self._field_sets.items():
            perm_map = field_set.coalesce(self._error_on_interference)
            for key, fields in perm_map.items():
                req_idx = len(self._requirements)
                req = RegionReq(region, *key)
                for field_id in fields:
                    self._requirement_map[(req, field_id)] = req_idx
                self._requirements.append((req, fields))

    def get_requirement_index(self, req, field_id):
        try:
            return self._requirement_map[(req, field_id)]
        except KeyError:
            req = req.promote_to_read_write()
            return self._requirement_map[(req, field_id)]


class OutputAnalyzer(object):
    def __init__(self, runtime):
        self._runtime = runtime
        self._groups = {}
        self._requirements = []
        self._requirement_map = {}

    @property
    def requirements(self):
        return self._requirements

    @property
    def empty(self):
        return len(self._groups) == 0

    def __del__(self):
        self._groups.clear()
        self._requirements.clear()
        self._requirement_map.clear()

    def insert(self, req, field_id, store):
        group = self._groups.get(req)
        if group is None:
            group = set()
            self._groups[req] = group
        group.add((field_id, store))

    def analyze_requirements(self):
        for req, group in self._groups.items():
            req_idx = len(self._requirements)
            fields = []
            field_set = set()
            for field_id, store in group:
                self._requirement_map[(req, field_id)] = req_idx
                if field_id in field_set:
                    raise RuntimeError(
                        f"{field_id} is duplicated in output requirement {req}"
                    )
                fields.append(field_id)
                field_set.add(field_id)

            self._requirements.append((req, fields))

    def get_requirement_index(self, req, field_id):
        return self._requirement_map[(req, field_id)]

    def update_storages(self):
        for req, group in self._groups.items():
            for field_id, store in group:
                req.update_storage(store, field_id)


class TaskLauncher(object):
    def __init__(
        self, context, task_id, mapper_id=0, tag=0, error_on_interference=True
    ):
        assert type(tag) != bool
        self._context = context
        self._runtime = context.runtime
        self._core_types = self._runtime.core_context.type_system
        self._task_id = task_id
        self._mapper_id = mapper_id
        self._inputs = []
        self._outputs = []
        self._reductions = []
        self._scalars = []
        self._req_analyzer = RequirementAnalyzer(error_on_interference)
        self._out_analyzer = OutputAnalyzer(context.runtime)
        self._future_args = list()
        self._future_map_args = list()
        self._tag = tag
        self._sharding_space = None
        self._point = None
        self._output_regions = list()
        self._error_on_interference = error_on_interference

    @property
    def library_task_id(self):
        return self._task_id

    @property
    def library_mapper_id(self):
        return self._mapper_id

    @property
    def legion_task_id(self):
        return self._context.get_task_id(self._task_id)

    @property
    def legion_mapper_id(self):
        return self._context.get_mapper_id(self._mapper_id)

    def __del__(self):
        del self._req_analyzer
        del self._out_analyzer
        self._future_args.clear()
        self._future_map_args.clear()
        self._output_regions.clear()

    def add_scalar_arg(self, value, dtype, untyped=True):
        self._scalars.append(
            ScalarArg(self._core_types, value, dtype, untyped)
        )

    def add_store(self, args, store, proj, perm, tag, flags):
        if store.kind is Future:
            has_storage = perm != Permission.WRITE
            read_only = perm == Permission.READ
            if has_storage:
                self.add_future(store.storage)
            args.append(FutureStoreArg(store, read_only, has_storage))

        else:
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
                    -1 if proj.redop is None else proj.redop,
                )
            )

    def add_input(self, store, proj, tag=0, flags=0):
        self.add_store(self._inputs, store, proj, Permission.READ, tag, flags)

    def add_output(self, store, proj, tag=0, flags=0):
        self.add_store(
            self._outputs, store, proj, Permission.WRITE, tag, flags
        )

    def add_reduction(self, store, proj, tag=0, flags=0, read_write=False):
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

    def add_unbound_output(self, store, fspace, field_id):
        req = OutputReq(self._runtime, fspace)

        self._out_analyzer.insert(req, field_id, store)

        self._outputs.append(
            RegionFieldArg(
                self._out_analyzer,
                store,
                1,
                req,
                field_id,
                -1,
            )
        )

    def add_future(self, future):
        self._future_args.append(future)

    def add_future_map(self, future_map):
        self._future_map_args.append(future_map)

    def set_sharding_space(self, space):
        self._sharding_space = space

    def set_point(self, point):
        self._point = point

    @staticmethod
    def pack_args(argbuf, args):
        argbuf.pack_32bit_uint(len(args))
        for arg in args:
            arg.pack(argbuf)

    def build_task(self, launch_domain, argbuf):
        self._req_analyzer.analyze_requirements()
        self._out_analyzer.analyze_requirements()

        self.pack_args(argbuf, self._inputs)
        self.pack_args(argbuf, self._outputs)
        self.pack_args(argbuf, self._reductions)
        self.pack_args(argbuf, self._scalars)

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
        for (out_req, fields) in self._out_analyzer.requirements:
            out_req.add(task, fields)
        for future_map in self._future_map_args:
            task.add_point_future(ArgumentMap(future_map=future_map))
        return task

    def build_single_task(self, argbuf):
        self._req_analyzer.analyze_requirements()
        self._out_analyzer.analyze_requirements()

        self.pack_args(argbuf, self._inputs)
        self.pack_args(argbuf, self._outputs)
        self.pack_args(argbuf, self._reductions)
        self.pack_args(argbuf, self._scalars)

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
        if self._req_analyzer.empty and self._out_analyzer.empty:
            task.set_local_function(True)
        if self._sharding_space is not None:
            task.set_sharding_space(self._sharding_space)
        if self._point is not None:
            task.set_point(self._point)
        return task

    def execute(self, launch_domain, redop=None):
        # Note that we should hold a reference to this buffer
        # until we launch a task, otherwise the Python GC will
        # collect the Python object holding the buffer, which
        # in turn will deallocate the C side buffer.
        argbuf = BufferBuilder()
        task = self.build_task(launch_domain, argbuf)
        if redop is not None:
            result = self._context.dispatch(task, redop=redop)
        else:
            result = self._context.dispatch(task)

        self._out_analyzer.update_storages()

        return result

    def execute_single(self):
        argbuf = BufferBuilder()
        result = self._context.dispatch(self.build_single_task(argbuf))
        self._out_analyzer.update_storages()
        return result


class CopyLauncher(object):
    def __init__(self, context, mapper_id=0, tag=0):
        assert type(tag) != bool
        self._context = context
        self._runtime = context.runtime
        self._mapper_id = mapper_id
        self._req_analyzer = RequirementAnalyzer()
        self._tag = tag
        self._sharding_space = None
        self._point = None
        self._output_regions = list()

    @property
    def library_mapper_id(self):
        return self._mapper_id

    @property
    def legion_mapper_id(self):
        return self._context.get_mapper_id(self._mapper_id)

    def __del__(self):
        del self._req_analyzer

    def add_store(self, store, proj, perm, tag, flags):
        assert store.kind is not Future
        assert store._transform is None

        region = store.storage.region
        field_id = store.storage.field.field_id

        req = RegionReq(region, perm, proj, tag, flags)

        self._req_analyzer.insert(req, field_id)

    def add_input(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.READ, tag, flags)

    def add_output(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.WRITE, tag, flags)

    def add_reduction(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.REDUCTION, tag, flags)

    def add_source_indirect(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.SOURCE_INDIRECT, tag, flags)

    def add_target_indirect(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.TARGET_INDIRECT, tag, flags)

    def set_sharding_space(self, space):
        self._sharding_space = space

    def set_point(self, point):
        self._point = point

    def build_copy(self, launch_domain):
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
        if self._point is not None:
            copy.set_point(self._point)
        return copy

    def build_single_copy(self):
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
        return copy

    def execute(self, launch_domain, redop=None):
        copy = self.build_copy(launch_domain)
        return self._context.dispatch(copy)

    def execute_single(self):
        copy = self.build_single_copy()
        return self._context.dispatch(copy)
