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

import legate.core.types as ty

from .launcher import CopyLauncher, TaskLauncher
from .legion import Future
from .solver import EqClass
from .store import Store


class Operation(object):
    def __init__(self, context, mapper_id=0):
        self._context = context
        self._mapper_id = mapper_id
        self._inputs = []
        self._outputs = []
        self._reductions = []
        self._temps = []
        self._future_output = None
        self._future_reduction = None
        self._constraints = EqClass()
        self._broadcasts = set()
        self._is_fused = False

    @property
    def context(self):
        return self._context

    @property
    def mapper_id(self):
        return self._mapper_id

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def reductions(self):
        return self._reductions

    @property
    def constraints(self):
        return self._constraints

    @property
    def broadcasts(self):
        return self._broadcasts

    def get_all_stores(self):
        stores = (
            set(self._inputs)
            | set(self._outputs)
            | set(store for (store, _) in self._reductions)
        )
        return stores

    @staticmethod
    def _check_store(store):
        if not isinstance(store, Store):
            raise ValueError(f"Expected a Store object, but got {type(store)}")

    def add_input(self, store):
        self._check_store(store)
        self._inputs.append(store)

    @property
    def _has_future_output(self):
        return (
            self._future_reduction is not None
            or self._future_output is not None
        )

    def _check_future_output(self):
        if self._has_future_output:
            raise ValueError(
                "Only one Future-backed store can be used for output"
            )

    def add_output(self, store):
        self._check_store(store)
        if store.kind is Future:
            self._check_future_output()
            self._future_output = store
        else:
            self._outputs.append(store)

    def add_temp(self, store):
        self._check_store(store)
        self._temps.append(store) #this may not be necessary

    def add_output(self, store):
        self._check_store(store)
        self._outputs.append(store)

    def add_reduction(self, store, redop):
        self._check_store(store)
        if store.kind is Future:
            self._check_future_output()
            self._future_reduction = (store, redop)
        else:
            self._reductions.append((store, redop))

    def add_alignment(self, store1, store2):
        self._check_store(store1)
        self._check_store(store2)
        if store1.shape != store2.shape:
            raise ValueError(
                "Stores must have the same shape to be aligned, "
                f"but got {store1.shape} and {store2.shape}"
            )
        self._constraints.record(store1, store2)

    def add_broadcast(self, store):
        self._broadcasts.add(store)

    def execute(self):
        self._context.runtime.submit(self)


class Task(Operation):
    def __init__(self, context, task_id, mapper_id=0):
        Operation.__init__(self, context, mapper_id)
        self._task_id = task_id
        self._scalar_args = []
        self._futures = []
        self._fusion_metadata = None

    def add_scalar_arg(self, value, dtype):
        self._scalar_args.append((value, dtype))

    def add_dtype_arg(self, dtype):
        code = self._context.type_system[dtype].code
        self._scalar_args.append((code, ty.int32))

    def add_future(self, future):
        self._futures.append(future)

    def add_fusion_metadata(self, fusion_metadata):
        self._is_fused = True
        self._fusion_metadata = fusion_metadata

    def launch(self, strategy):
        launcher = TaskLauncher(self.context, self._task_id, self.mapper_id)

        if self._is_fused:
            launcher.add_fusion_metadata(self._is_fused, self._fusion_metadata)
 
        for input in self._inputs:
            proj = strategy.get_projection(input)
            launcher.add_input(input, proj)
        for temp in self._temps:
            proj = strategy.get_projection(temp)
            launcher.add_temp(temp, proj)
            partition = strategy.get_partition(temp)
            # We update the key partition of a store only when it gets updated
            temp.set_key_partition(partition)
        for output in self._outputs:
            if output.unbound:
                continue
            proj = strategy.get_projection(output)
            launcher.add_output(output, proj)
            partition = strategy.get_partition(output)
            # We update the key partition of a store only when it gets updated
            output.set_key_partition(partition)
        for (reduction, redop) in self._reductions:
            partition = strategy.get_partition(reduction)
            can_read_write = partition.is_disjoint_for(strategy, reduction)
            proj = strategy.get_projection(reduction)
            proj.redop = redop
            launcher.add_reduction(reduction, proj, read_write=can_read_write)
        for output in self._outputs:
            if not output.unbound:
                continue
            fspace = strategy.get_field_space(output)
            field_id = fspace.allocate_field(output.type)
            launcher.add_unbound_output(output, fspace, field_id)

        for (arg, dtype) in self._scalar_args:
            launcher.add_scalar_arg(arg, dtype)

        for future in self._futures:
            launcher.add_future(future)

        if self._future_output is not None:
            strategy.launch(launcher, self._future_output)
        elif self._future_reduction is not None:
            (store, redop) = self._future_reduction
            strategy.launch(launcher, store, redop)
        else:
            strategy.launch(launcher)


class Copy(Operation):
    def __init__(self, context, mapper_id=0):
        Operation.__init__(self, context, mapper_id)
        self._source_indirects = []
        self._target_indirects = []

    @property
    def inputs(self):
        return (
            super(Copy, self).inputs
            + self._source_indirects
            + self._target_indirects
        )

    def add_source_indirect(self, store):
        self._source_indirects.append(store)

    def add_target_indirect(self, store):
        self._target_indirects.append(store)

    def launch(self, strategy):
        launcher = CopyLauncher(self.context, self.mapper_id)

        assert len(self._inputs) == len(self._outputs) or len(
            self._inputs
        ) == len(self._reductions)
        assert len(self._source_indirects) == 0 or len(
            self._source_indirects
        ) == len(self._inputs)
        assert len(self._target_indirects) == 0 or len(
            self._target_indirects
        ) == len(self._outputs)

        for input in self._inputs:
            proj = strategy.get_projection(input)
            launcher.add_input(input, proj)
        for output in self._outputs:
            assert not output.unbound
            proj = strategy.get_projection(output)
            launcher.add_output(output, proj)
        for indirect in self._source_indirects:
            proj = strategy.get_projection(indirect)
            launcher.add_source_indirect(indirect, proj)
        for indirect in self._target_indirects:
            proj = strategy.get_projection(indirect)
            launcher.add_target_indirect(indirect, proj)

        for (reduction, redop) in self._reductions:
            proj = strategy.get_projection(reduction)
            proj.redop = redop
            launcher.add_reduction(reduction, proj)

        strategy.launch(launcher)
