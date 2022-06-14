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

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union, cast

import numpy as np

from . import Future, legion
from .operation import AutoTask, Copy, ManualTask, Reduce
from .resource import ResourceScope
from .types import TypeSystem

if TYPE_CHECKING:
    import numpy.typing as npt
    from pyarrow import DataType

    from . import ArgumentMap, Rect
    from ._legion.util import Dispatchable
    from .communicator import Communicator
    from .legate import Library
    from .runtime import Runtime
    from .shape import Shape
    from .store import RegionField, Store

T = TypeVar("T")


class Context:
    def __init__(
        self,
        runtime: Runtime,
        library: Library,
        inherit_core_types: bool = True,
    ) -> None:
        """
        A Context is a named scope for Legion resources used in a Legate
        library. A Context is created when the library is registered
        for the first time to the Legate runtime, and it must be passed
        when the library registers or makes accesses to its Legion resources.
        Resources that are scoped locally to each library include
        task ids, projection and sharding functor ids, and reduction operator
        ids.
        """
        self._runtime = runtime
        self._library = library
        self._type_system = TypeSystem(inherit_core_types)

        config = library.get_resource_configuration()
        name = library.get_name().encode("utf-8")
        lg_runtime = self._runtime.legion_runtime

        def _create_scope(
            api: Any, category: str, max_counts: int
        ) -> ResourceScope:
            base = (
                api(lg_runtime, name, max_counts) if max_counts > 0 else None
            )
            return ResourceScope(self, base, category)

        self._task_scope = _create_scope(
            legion.legion_runtime_generate_library_task_ids,
            "task",
            config.max_tasks,
        )
        self._mapper_scope = _create_scope(
            legion.legion_runtime_generate_library_mapper_ids,
            "mapper",
            config.max_mappers,
        )
        self._redop_scope = _create_scope(
            legion.legion_runtime_generate_library_reduction_ids,
            "reduction op",
            config.max_reduction_ops,
        )
        self._proj_scope = _create_scope(
            legion.legion_runtime_generate_library_projection_ids,
            "Projection functor",
            config.max_projections,
        )
        self._shard_scope = _create_scope(
            legion.legion_runtime_generate_library_sharding_ids,
            "sharding functor",
            config.max_shardings,
        )

        self._unique_op_id = 0

    def destroy(self) -> None:
        self._library.destroy()

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def library(self) -> Library:
        return self._library

    @property
    def core_library(self) -> Any:
        return self._runtime.core_library

    @property
    def first_mapper_id(self) -> Union[int, None]:
        return self._mapper_scope._base

    @property
    def first_redop_id(self) -> Union[int, None]:
        return self._redop_scope._base

    @property
    def first_shard_id(self) -> Union[int, None]:
        return self._shard_scope._base

    @property
    def empty_argmap(self) -> ArgumentMap:
        return self._runtime.empty_argmap

    @property
    def type_system(self) -> TypeSystem:
        return self._type_system

    def get_task_id(self, task_id: int) -> int:
        return self._task_scope.translate(task_id)

    @property
    def mapper_id(self) -> int:
        return self.get_mapper_id(0)

    def get_mapper_id(self, mapper_id: int) -> int:
        return self._mapper_scope.translate(mapper_id)

    def get_reduction_op_id(self, redop_id: int) -> int:
        return self._redop_scope.translate(redop_id)

    def get_projection_id(self, proj_id: int) -> int:
        if proj_id == 0:
            return proj_id
        else:
            return self._proj_scope.translate(proj_id)

    def get_sharding_id(self, shard_id: int) -> int:
        return self._shard_scope.translate(shard_id)

    def get_tunable(
        self, tunable_id: int, dtype: DataType, mapper_id: int = 0
    ) -> npt.NDArray[Any]:
        dt = np.dtype(dtype.to_pandas_dtype())
        mapper_id = self.get_mapper_id(mapper_id)
        fut = Future(
            legion.legion_runtime_select_tunable_value(
                self._runtime.legion_runtime,
                self._runtime.legion_context,
                tunable_id,
                mapper_id,
                0,
            )
        )
        buf = fut.get_buffer(dt.itemsize)
        return np.frombuffer(buf, dtype=dt)[0]

    def get_unique_op_id(self) -> int:
        return self._runtime.get_unique_op_id()

    def create_task(
        self,
        task_id: int,
        mapper_id: int = 0,
        manual: Optional[bool] = False,
        launch_domain: Optional[Rect] = None,
    ) -> Union[AutoTask, ManualTask]:
        unique_op_id = self.get_unique_op_id()
        if not manual:
            return AutoTask(self, task_id, mapper_id, unique_op_id)
        else:
            if launch_domain is None:
                raise RuntimeError(
                    "Launch domain must be specified for "
                    "manual parallelization"
                )
            return ManualTask(
                self,
                task_id,
                launch_domain,
                mapper_id,
                unique_op_id,
            )

    def create_manual_task(
        self,
        task_id: int,
        mapper_id: int = 0,
        launch_domain: Optional[Rect] = None,
    ) -> ManualTask:
        return cast(
            ManualTask,
            self.create_task(
                task_id=task_id,
                mapper_id=mapper_id,
                manual=True,
                launch_domain=launch_domain,
            ),
        )

    def create_auto_task(
        self,
        task_id: int,
        mapper_id: int = 0,
        launch_domain: Optional[Rect] = None,
    ) -> AutoTask:
        return cast(
            AutoTask,
            self.create_task(
                task_id=task_id,
                mapper_id=mapper_id,
                manual=False,
                launch_domain=launch_domain,
            ),
        )

    def create_copy(self, mapper_id: int = 0) -> Copy:
        return Copy(self, mapper_id)

    def dispatch(self, op: Dispatchable[T]) -> T:
        return self._runtime.dispatch(op)

    def dispatch_single(self, op: Dispatchable[T]) -> T:
        return self._runtime.dispatch_single(op)

    def create_store(
        self,
        ty: Any,
        shape: Optional[Shape] = None,
        storage: Optional[Union[RegionField, Future]] = None,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> Store:
        dtype = self.type_system[ty]
        return self._runtime.create_store(
            dtype,
            shape=shape,
            data=storage,
            optimize_scalar=optimize_scalar,
            ndim=ndim,
        )

    def get_nccl_communicator(self) -> Communicator:
        return self._runtime.get_nccl_communicator()

    def get_cpu_communicator(self) -> Communicator:
        return self._runtime.get_cpu_communicator()

    def issue_execution_fence(self, block: bool = False) -> None:
        self._runtime.issue_execution_fence(block=block)

    def tree_reduce(
        self, task_id: int, store: Store, mapper_id: int = 0, radix: int = 4
    ) -> Store:
        result = self.create_store(store.type)
        unique_op_id = self.get_unique_op_id()

        # Make sure we flush the scheduling window, as we will bypass
        # the partitioner below
        self.runtime.flush_scheduling_window()

        # A single Reduce operation is mapepd to a whole reduction tree
        task = Reduce(self, task_id, radix, mapper_id, unique_op_id)
        task.add_input(store)
        task.add_output(result)
        task.execute()
        return result
