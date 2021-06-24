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

from .launcher import TaskLauncher
from .solver import EqClass, Partitioner


class Operation(object):
    def __init__(self, context, mapper_id=0):
        self._context = context
        self._mapper_id = mapper_id
        self._no_accesses = []
        self._inputs = []
        self._outputs = []
        self._reductions = []
        self._constraints = EqClass()

    @property
    def context(self):
        return self._context

    @property
    def mapper_id(self):
        return self._mapper_id

    @property
    def no_accesses(self):
        return self._no_accesses

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

    def get_all_stores(self):
        stores = (
            set(self._no_accesses)
            | set(self._inputs)
            | set(self._outputs)
            | set(store for (store, _) in self._reductions)
        )
        return stores

    def add_no_access(self, store):
        self._no_accesses.append(store)

    def add_input(self, store):
        self._inputs.append(store)

    def add_output(self, store):
        self._outputs.append(store)

    def add_reduction(self, store, redop):
        self._reductions.append((store, redop))

    def add_alignment(self, store1, store2):
        if store1.shape != store2.shape:
            raise ValueError(
                "Stores must have the same shape to be aligned, "
                f"but got {store1.shape} and {store2.shape}"
            )
        self._constraints.record(store1, store2)


class Task(Operation):
    def __init__(self, context, task_id, mapper_id=0):
        Operation.__init__(self, context, mapper_id)
        self._task_id = task_id
        self._scalar_args = []

    def add_scalar_arg(self, value, dtype):
        self._scalar_args.append((value, dtype))

    def add_dtype_arg(self, dtype):
        code = self._context.type_system[dtype].code
        self._scalar_args.append((code, ty.int32))

    def execute(self):
        partitioner = Partitioner(self._context.runtime, [self])
        strategy = partitioner.partition_stores()

        launcher = TaskLauncher(self.context, self._task_id, self.mapper_id)

        for no_access in self._no_accesses:
            launcher.add_no_access(no_access, strategy[no_access])
        for input in self._inputs:
            launcher.add_input(input, strategy[input])
        for output in self._outputs:
            launcher.add_output(output, strategy[output])
        for (reduction, redop) in self._reductions:
            partition = strategy[reduction]
            partition.redop = redop
            launcher.add_reduction(output, partition)
        for (arg, dtype) in self._scalar_args:
            launcher.add_scalar_arg(arg, dtype)

        strategy.launch(launcher)
