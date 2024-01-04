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

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import numpy as np

from . import Future, legion
from ._legion.util import Logger
from ._lib.context import (  # type: ignore[import-not-found]
    Context as CppContext,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from . import ArgumentMap, Rect
    from ._legion.util import Dispatchable
    from .communicator import Communicator
    from .legate import Library
    from .machine import Machine
    from .operation import AutoTask, Copy, ManualTask
    from .runtime import Runtime
    from .shape import Shape
    from .store import RegionField, Store
    from .types import Dtype

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

        name = library.get_name()

        self._cpp_context = CppContext(name, False)

        self._libname = library.get_name()
        self._logger = Logger(library.get_name())

        self._mapper_id = self._cpp_context.get_mapper_id()

    def destroy(self) -> None:
        self._logger.destroy()
        self._library.destroy()

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def runtime(self) -> Runtime:
        """
        Returns the runtime

        Returns
        -------
        Runtime
            The runtime object
        """
        return self._runtime

    @property
    def library(self) -> Library:
        return self._library

    @property
    def core_library(self) -> Any:
        return self._runtime.core_library

    @property
    def first_redop_id(self) -> Union[int, None]:
        return self.get_reduction_op_id(0)

    @property
    def first_shard_id(self) -> Union[int, None]:
        return self.get_sharding_id(0)

    @property
    def empty_argmap(self) -> ArgumentMap:
        return self._runtime.empty_argmap

    def get_task_id(self, task_id: int) -> int:
        return self._cpp_context.get_task_id(task_id)

    @property
    def mapper_id(self) -> int:
        return self._mapper_id

    def get_reduction_op_id(self, redop_id: int) -> int:
        return self._cpp_context.get_reduction_op_id(redop_id)

    def get_projection_id(self, proj_id: int) -> int:
        if proj_id == 0:
            return proj_id
        else:
            return self._cpp_context.get_projection_id(proj_id)

    def get_sharding_id(self, shard_id: int) -> int:
        return self._cpp_context.get_sharding_id(shard_id)

    def get_tunable(self, tunable_id: int, dtype: Dtype) -> npt.NDArray[Any]:
        """
        Queries a tunable parameter to the mapper.

        Parameters
        ----------
        tunable_id : int
            Tunable id. Local to each mapper.

        dtype : Dtype
            Value type

        Returns
        -------
        np.ndarray
            A NumPy array holding the value of the tunable parameter
        """
        dt = dtype.to_numpy_dtype()
        fut = Future(
            legion.legion_runtime_select_tunable_value(
                self._runtime.legion_runtime,
                self._runtime.legion_context,
                tunable_id,
                self.mapper_id,
                0,
            )
        )
        buf = fut.get_buffer(dt.itemsize)
        return np.frombuffer(buf, dtype=dt)[0]

    def get_unique_op_id(self) -> int:
        return self._runtime.get_unique_op_id()

    def slice_machine_for_task(self, task_id: int) -> Machine:
        """
        Narrows down the current machine by cutting out processors
        for which the task has no variant
        """
        task_info = self._cpp_context.find_task(task_id)
        if not task_info.valid:
            raise ValueError(
                f"Library '{self._libname}' does not have task {task_id}"
            )

        machine = self._runtime.machine.filter_ranges(
            task_info, self._runtime.variant_ids
        )

        if machine.empty:
            error_msg = (
                f"Task {task_id} ({task_info.name}) of library "
                f"'{self._libname}' does not have any valid variant for "
                "the current machine configuration."
            )
            raise ValueError(error_msg)

        return machine

    def create_manual_task(
        self,
        task_id: int,
        launch_domain: Rect,
    ) -> ManualTask:
        """
        Creates a manual task.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Parameters
        ----------
        task_id : int
            Task id. Scoped locally within the context; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        launch_domain : Rect, optional
            Launch domain of the task.

        Returns
        -------
        ManualTask
            A new task
        """
        return self._runtime.create_manual_task(self, task_id, launch_domain)

    def create_auto_task(
        self,
        task_id: int,
    ) -> AutoTask:
        """
        Creates an auto task.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Parameters
        ----------
        task_id : int
            Task id. Scoped locally within the context; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        Returns
        -------
        AutoTask
            A new automatically parallelized task

        See Also
        --------
        Context.create_task
        """
        return self._runtime.create_auto_task(self, task_id)

    def create_copy(self) -> Copy:
        """
        Creates a copy operation.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Returns
        -------
        Copy
            A new copy operation
        """

        return self._runtime.create_copy()

    def issue_fill(
        self,
        lhs: Store,
        value: Store,
    ) -> None:
        """
        Fills the store with a constant value.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Parameters
        ----------
        lhs : Store
            Store to fill

        value : Store
            Store holding the constant value to fill the ``lhs`` with

        Raises
        ------
        ValueError
            If the ``value`` is not scalar or the ``lhs`` is either unbound or
            scalar
        """
        self._runtime.issue_fill(lhs, value)

    def dispatch(self, op: Dispatchable[T]) -> T:
        return self._runtime.dispatch(op)

    def dispatch_single(self, op: Dispatchable[T]) -> T:
        return self._runtime.dispatch_single(op)

    def create_store(
        self,
        dtype: Dtype,
        shape: Optional[Union[Shape, tuple[int, ...]]] = None,
        storage: Optional[Union[RegionField, Future]] = None,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> Store:
        """
        Creates a fresh store.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Parameters
        ----------
        dtype : Dtype
            Type of the elements

        shape : Shape or tuple[int], optional
            Shape of the store. The store becomes unbound if no shape is
            given.

        storage : RegionField or Future, optional
            Optional storage to initialize the store with. Used only when the
            store is constructed from a future holding a scalar value.

        optimize_scalar : bool
            If ``True``, the runtime will use a ``Future`` when the store's
            size is 1

        ndim : int, optional
            Dimension of the store. Must be passed if the store is unbound.

        Returns
        -------
        Store
            A new store
        """
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

    @property
    def has_cpu_communicator(self) -> bool:
        return self._runtime.has_cpu_communicator

    def issue_execution_fence(self, block: bool = False) -> None:
        """
        Issues an execution fence. A fence is a special operation that
        guarantees that all upstream operations finish before any of the
        downstream operations start. The caller can optionally block on
        completion of all upstream operations.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Parameters
        ----------
        block : bool
            If ``True``, the call blocks until all upstream operations finish.
        """
        self._runtime.issue_execution_fence(block=block)

    def tree_reduce(self, task_id: int, store: Store, radix: int = 4) -> Store:
        """
        Performs a user-defined reduction by building a tree of reduction
        tasks. At each step, the reducer task gets up to ``radix`` input stores
        and is supposed to produce outputs in a single unbound store.

        .. deprecated:: 23.7.0
            This method is an alias to the one defined in ``Runtime`` and will
            be removed in later releases.

        Parameters
        ----------
        task_id : int
            Id of the reducer task

        store : Store
            Store to perform reductions on

        radix : int
            Fan-in of each reducer task. If the store is partitioned into
            :math:`N` sub-stores by the runtime, then the first level of
            reduction tree has :math:`\\ceil{N / \\mathtt{radix}}` reducer
            tasks.

        Returns
        -------
        Store
            Store that contains reduction results
        """
        return self._runtime.tree_reduce(self, task_id, store, radix)
