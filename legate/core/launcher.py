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

from enum import IntEnum

import legate.core.types as ty

from .legion import ArgumentMap, BufferBuilder, IndexTask, Task as SingleTask


class Permission(IntEnum):
    NO_ACCESS = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    REDUCTION = 4


class ScalarArg(object):
    _serializers = {
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

    def __init__(self, value, dtype):
        self._value = value
        self._dtype = dtype

    def pack(self, buf):
        if isinstance(self._dtype, tuple) or isinstance(self._dtype, list):
            assert len(self._dtype) == 1
            dtype = self._dtype[0]
            self._serializers[ty.int32](buf, len(self._value))
            serializer = self._serializers[dtype]
            for value in self._value:
                serializer(buf, value)
        elif self._dtype in self._serializers:
            self._serializers[self._dtype](buf, self._value)
        else:
            raise ValueError("Unsupported data type: %s" % str(self._dtype))

    def __str__(self):
        return f"ScalarArg({self._value}, {self._dtype})"

    def __repr__(self):
        return str(self)


class DtypeArg(object):
    def __init__(self, dtype):
        self._dtype = dtype

    def pack(self, buf):
        buf.pack_dtype(self._dtype)

    def __str__(self):
        return f"DtypeArg({self._dtype})"

    def __repr__(self):
        return str(self)


class RegionFieldArg(object):
    def __init__(self, op, dim, key, field_id):
        self._op = op
        self._dim = dim
        self._key = key
        self._field_id = field_id

    def pack(self, buf):
        buf.pack_32bit_int(self._dim)
        buf.pack_32bit_uint(
            self._op.get_requirement_index(self._key, self._field_id)
        )
        buf.pack_32bit_uint(self._field_id)

    def __str__(self):
        return f"RegionFieldArg({self._dim}, {self._field_id})"

    def __repr__(self):
        return str(self)


class OutputRegionArg(object):
    def __init__(self, idx, field_id):
        self._idx = idx
        self._field_id = field_id

    def pack(self, buf):
        buf.pack_32bit_uint(self._idx)
        buf.pack_32bit_uint(self._field_id)

    def __str__(self):
        return f"OutputRegionArg({self._idx}, {self._field_id})"

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


class Broadcast(object):
    def __init__(self, redop=None):
        self.redop = redop

    def add(self, task, req, fields):
        f = _index_task_calls[req.permission]
        if self.redop is None:
            f(task, req.region, fields, 0, parent=req.region, tag=req.tag)
        else:
            assert req.permission == Permission.REDUCTION
            f(
                task,
                req.region,
                fields,
                self.redop,
                0,
                parent=req.region,
                tag=req.tag,
            )

    def add_single(self, task, req, fields):
        f = _single_task_calls[req.permission]
        if self.redop is None:
            f(task, req.region, fields, tag=req.tag, flags=req.flags)
        else:
            assert req.permission == Permission.REDUCTION
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
    def __init__(self, part, proj=0, redop=None):
        self.part = part
        self.proj = proj
        self.redop = redop

    def add(self, task, req, fields):
        f = _index_task_calls[req.permission]
        if self.redop is None:
            f(task, self.part, fields, self.proj, tag=req.tag, flags=req.flags)
        else:
            assert req.permission == Permission.REDUCTION
            f(
                task,
                self.part,
                fields,
                self.redop,
                self.proj,
                tag=req.tag,
                flags=req.flags,
            )

    def add_single(self, task, req, fields):
        f = _single_task_calls[req.permission]
        if self.redop is None:
            f(task, req.region, fields, tag=req.tag)
        else:
            assert req.permission == Permission.REDUCTION
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

    def __repr__(self):
        return repr(
            (self.region, self.permission, self.proj, self.tag, self.flags)
        )

    def __hash__(self):
        return hash(
            (self.region, self.proj, self.permission, self.tag, self.flags)
        )

    def __eq__(self, other):
        return (
            self.region == other.region
            and self.proj == other.proj
            and self.permission == other.permission
            and self.tag == other.tag
            and self.flags == other.flags
        )


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

    def coalesce(self):
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
                raise ValueError(
                    f"Interfering requirements found: {all_entries}"
                )

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

    def coalesce(self):
        coalesced = {}
        for field_id, proj_set in self._fields.items():
            proj_infos = proj_set.coalesce()
            for key in proj_infos:
                if key in coalesced:
                    coalesced[key].append(field_id)
                else:
                    coalesced[key] = [field_id]

        return coalesced


class TaskLauncher(object):
    def __init__(self, context, task_id, mapper_id=0, tag=0):
        assert type(tag) != bool
        self._context = context
        self._runtime = context.runtime
        self._task_id = task_id
        self._mapper_id = mapper_id
        self._args = list()
        self._region_args = {}
        self._region_reqs = list()
        self._region_reqs_indices = {}
        self._next_region_idx = 0
        self._projections = list()
        self._future_args = list()
        self._future_map_args = list()
        self._tag = tag
        self._sharding_space = None
        self._point = None
        self._output_regions = list()

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
        self._region_args.clear()
        self._projections.clear()
        self._future_args.clear()
        self._future_map_args.clear()

    def _coalesce_region_requirements(self):
        for region, field_set in self._region_args.items():
            perm_map = field_set.coalesce()
            for key, fields in perm_map.items():
                req_idx = len(self._region_reqs)
                req = RegionReq(region, *key)
                for field_id in fields:
                    self._region_reqs_indices[(req, field_id)] = req_idx
                self._region_reqs.append((req, fields))

    def add_scalar_arg(self, value, dtype):
        self._args.append(ScalarArg(value, dtype))

    def get_requirement_index(self, key, field_id):
        try:
            return self._region_reqs_indices[(key, field_id)]
        except KeyError:
            key = RegionReq(
                key.region, Permission.READ_WRITE, key.proj, key.tag, key.flags
            )
            return self._region_reqs_indices[(key, field_id)]

    def add_store(self, store, proj, perm, tag, flags):
        store.serialize(self)
        redop = -1 if proj.redop is None else proj.redop
        self.add_scalar_arg(redop, ty.int32)

        if store.scalar:
            if perm != Permission.READ:
                raise ValueError("Scalar stores must be read only")
            self.add_future(store.storage)
            self.add_scalar_arg(store.get_root().shape, (ty.int64,))
            return

        region = store.storage.region
        field_id = store.storage.field.field_id

        if region in self._region_args:
            field_set = self._region_args[region]
        else:
            field_set = FieldSet()
            self._region_args[region] = field_set
        proj_info = (proj, tag, flags)
        field_set.insert(field_id, perm, proj_info)

        self._args.append(
            RegionFieldArg(
                self,
                region.index_space.get_dim(),
                RegionReq(region, perm, *proj_info),
                field_id,
            )
        )

    def add_no_access(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.NO_ACCESS, tag, flags)

    def add_input(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.READ, tag, flags)

    def add_output(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.WRITE, tag, flags)

    def add_inout(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.READ_WRITE, tag, flags)

    def add_reduction(self, store, proj, tag=0, flags=0):
        self.add_store(store, proj, Permission.REDUCTION, tag, flags)

    def add_unbound_output(self, store):
        region_field = self._runtime.allocate_unbound_field(store.get_dtype())
        output_region = self._runtime.create_output_region(region_field)

        store.serialize(self)
        self.add_scalar_arg(-1, ty.int32)

        idx = len(self._output_regions)
        self._output_regions.append((store, region_field, output_region))
        self._args.append(OutputRegionArg(idx, region_field.field.field_id))

    def add_future(self, future):
        self._future_args.append(future)

    def add_future_map(self, future_map):
        self._future_map_args.append(future_map)

    def set_sharding_space(self, space):
        self._sharding_space = space

    def set_point(self, point):
        self._point = point

    def build_task(self, launch_domain, argbuf):
        self._coalesce_region_requirements()

        for arg in self._args:
            arg.pack(argbuf)
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

        for (req, fields) in self._region_reqs:
            req.proj.add(task, req, fields)
        for future in self._future_args:
            task.add_future(future)
        for (_, _, output_region) in self._output_regions:
            task.add_output(output_region)
        for future_map in self._future_map_args:
            task.add_point_future(ArgumentMap(future_map=future_map))
        return task

    def build_single_task(self, argbuf):
        self._coalesce_region_requirements()

        for arg in self._args:
            arg.pack(argbuf)
        task = SingleTask(
            self.legion_task_id,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self.legion_mapper_id,
            tag=self._tag,
        )
        for (req, fields) in self._region_reqs:
            req.proj.add_single(task, req, fields)
        for future in self._future_args:
            task.add_future(future)
        for (_, _, output_region) in self._output_regions:
            task.add_output(output_region)
        if len(self._region_args) == 0:
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

        for (store, region_field, output_region) in self._output_regions:
            region = output_region.get_region()
            bounds = region.index_space.get_bounds()
            shape = tuple(bounds.hi[idx] + 1 for idx in range(bounds.dim))
            region_field.region = region
            region_field.shape = shape
            region_field.field.region = region
            region_field.field.shape = shape
            store.set_storage(region_field, shape=shape)

        return result

    def execute_single(self):
        argbuf = BufferBuilder()
        result = self._context.dispatch(self.build_single_task(argbuf))
        for (store, region_field, output_region) in self._output_regions:
            region = output_region.get_region()
            bounds = region.index_space.get_bounds()
            shape = tuple(bounds.hi[idx] + 1 for idx in range(bounds.dim))
            region_field.region = region
            region_field.shape = shape
            region_field.field.region = region
            region_field.field.shape = shape
            store.set_storage(region_field, shape=shape)
        return result
