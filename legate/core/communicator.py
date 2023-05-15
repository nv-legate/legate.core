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

import struct
from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING

from . import FutureMap, Point, Rect
from .machine import ProcessorKind

if TYPE_CHECKING:
    from .machine import ProcessorRange
    from .runtime import Runtime


class Communicator(ABC):
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._context = runtime.core_context

        self._handles: dict[tuple[int, ProcessorRange], FutureMap] = {}
        # From launch domains to communicator future maps transformed to N-D
        self._nd_handles: dict[tuple[Rect, ProcessorRange], FutureMap] = {}

    def _get_1d_handle(self, volume: int) -> FutureMap:
        proc_range = self._runtime.machine.get_processor_range()
        key = (volume, proc_range)
        comm = self._handles.get(key)
        if comm is not None:
            return comm
        comm = self._initialize(volume)
        self._handles[key] = comm
        return comm

    def _transform_handle(
        self, comm: FutureMap, launch_domain: Rect
    ) -> FutureMap:
        proc_range = self._runtime.machine.get_processor_range()
        key = (launch_domain, proc_range)
        match = self._nd_handles.get(key)
        if match is not None:
            return match
        match = self._runtime.delinearize_future_map(comm, launch_domain)
        self._nd_handles[key] = match
        return match

    def get_handle(self, launch_domain: Rect) -> FutureMap:
        comm = self._get_1d_handle(launch_domain.get_volume())
        if launch_domain.dim > 1:
            comm = self._transform_handle(comm, launch_domain)
        return comm

    def initialize(self, volume: int) -> None:
        self._get_1d_handle(volume)

    def destroy(self) -> None:
        for (volume, _), handle in self._handles.items():
            self._finalize(volume, handle)
        # Drop the references to the handles dict after
        # all handles have been finalized to ensure that
        # no references to FutureMaps are kept.
        self._handles = {}

    @abstractproperty
    def needs_barrier(self) -> bool:
        ...

    @abstractmethod
    def _initialize(self, volume: int) -> FutureMap:
        ...

    @abstractmethod
    def _finalize(self, volume: int, handle: FutureMap) -> None:
        ...


class NCCLCommunicator(Communicator):
    def __init__(self, runtime: Runtime) -> None:
        super().__init__(runtime)
        library = runtime.core_library

        self._init_nccl_id = library.LEGATE_CORE_INIT_NCCL_ID_TASK_ID
        self._init_nccl = library.LEGATE_CORE_INIT_NCCL_TASK_ID
        self._finalize_nccl = library.LEGATE_CORE_FINALIZE_NCCL_TASK_ID
        self._tag = library.LEGATE_GPU_VARIANT
        self._needs_barrier = runtime.nccl_needs_barrier

    @property
    def needs_barrier(self) -> bool:
        return self._needs_barrier

    def _initialize(self, volume: int) -> FutureMap:
        from .launcher import TaskLauncher as Task

        # This doesn't need to run on a GPU, but will use it anyway
        task = Task(
            self._context, self._init_nccl_id, tag=self._tag, side_effect=True
        )
        nccl_id = task.execute_single()

        task = Task(self._context, self._init_nccl, tag=self._tag)
        task.add_future(nccl_id)
        task.set_concurrent(True)
        handle = task.execute(Rect([volume])).future_map
        return handle

    def _finalize(self, volume: int, handle: FutureMap) -> None:
        from .launcher import TaskLauncher as Task

        task = Task(self._context, self._finalize_nccl, tag=self._tag)
        # Finalize may not need to be concurrent, but set it just in case
        task.set_concurrent(True)
        task.add_future_map(handle)
        task.execute(Rect([volume]))


class CPUCommunicator(Communicator):
    def __init__(self, runtime: Runtime) -> None:
        super().__init__(runtime)
        library = runtime.core_library

        self._init_cpucoll_mapping = (
            library.LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID
        )
        self._init_cpucoll = library.LEGATE_CORE_INIT_CPUCOLL_TASK_ID
        self._finalize_cpucoll = library.LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID
        self._needs_barrier = False

    def destroy(self) -> None:
        if len(self._handles) > 0:
            # Call the default destroy to finalize all cpu communicators that
            #   have been created
            Communicator.destroy(self)
            # We need to make sure all communicators are destroyed before
            #   finalize the cpu collective library.
            # However, this call is only required when there are cpu
            #   communicators created before
            self._runtime.issue_execution_fence(block=True)
        # Finalize the cpu collective library.
        # This call is always required because we always init the cpu
        #   collective library during legate library initialization
        self._runtime.core_library.legate_cpucoll_finalize()

    @property
    def needs_barrier(self) -> bool:
        return self._needs_barrier

    @property
    def _tag(self) -> int:
        if self._runtime.machine.count(ProcessorKind.OMP) > 0:
            return self._runtime.variant_ids[ProcessorKind.OMP]
        else:
            return self._runtime.variant_ids[ProcessorKind.CPU]

    def _initialize(self, volume: int) -> FutureMap:
        from .launcher import TaskLauncher as Task

        cpucoll_uid = self._runtime.core_library.legate_cpucoll_initcomm()
        buf = struct.pack("i", cpucoll_uid)
        cpucoll_uid_f = self._runtime.create_future(buf, len(buf))
        task = Task(self._context, self._init_cpucoll_mapping, tag=self._tag)
        mapping_table_fm = task.execute(Rect([volume])).future_map
        task = Task(self._context, self._init_cpucoll, tag=self._tag)
        task.add_future(cpucoll_uid_f)
        for i in range(volume):
            f = mapping_table_fm.get_future(Point([i]))
            task.add_future(f)
        task.set_concurrent(True)
        handle = task.execute(Rect([volume])).future_map
        return handle

    def _finalize(self, volume: int, handle: FutureMap) -> None:
        from .launcher import TaskLauncher as Task

        task = Task(self._context, self._finalize_cpucoll, tag=self._tag)
        task.add_future_map(handle)
        task.set_concurrent(True)
        task.execute(Rect([volume]))
        self._runtime.issue_execution_fence()
