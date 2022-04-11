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

from abc import ABC, abstractmethod

from . import Rect
from .launcher import TaskLauncher as Task


class Communicator(ABC):
    def __init__(self, runtime):
        self._runtime = runtime
        self._context = runtime.core_context

        self._comms = {}
        # From launch domains to communicator future maps transformed to N-D
        self._nd_comms = {}

    def _get_1d_communicator(self, volume):
        if volume in self._comms:
            return self._comms[volume]
        comm = self._initialize(volume)
        self._comms[volume] = comm
        return comm

    def _transform_communicator(self, comm, launch_domain):
        if launch_domain in self._nd_comms:
            return self._nd_comms[launch_domain]
        comm = self._runtime.delinearize_future_map(comm, launch_domain)
        self._nd_comms[launch_domain] = comm
        return comm

    def get_communicator(self, launch_domain):
        comm = self._get_1d_communicator(launch_domain.get_volume())
        if launch_domain.dim > 1:
            comm = self._transform_communicator(comm, launch_domain)
        return comm

    def destroy(self):
        for volume, handle in self._comms.items():
            self._finalize(volume, handle)

    @abstractmethod
    def _initialize(self, volume):
        ...

    @abstractmethod
    def _finalize(self, volume, handle):
        ...


class NCCLCommunicator(Communicator):
    def __init__(self, runtime):
        super(NCCLCommunicator, self).__init__(runtime)
        library = runtime.core_library

        self._init_nccl_id = library.LEGATE_CORE_INIT_NCCL_ID_TASK_ID
        self._init_nccl = library.LEGATE_CORE_INIT_NCCL_TASK_ID
        self._finalize_nccl = library.LEGATE_CORE_FINALIZE_NCCL_TASK_ID
        self._tag = library.LEGATE_GPU_VARIANT

    def _initialize(self, volume):
        # This doesn't need to run on a GPU, but will use it anyway
        task = Task(
            self._context, self._init_nccl_id, tag=self._tag, side_effect=True
        )
        nccl_id = task.execute_single()

        task = Task(self._context, self._init_nccl, tag=self._tag)
        task.add_future(nccl_id)
        handle = task.execute(Rect([volume]))
        self._runtime.issue_execution_fence()
        return handle

    def _finalize(self, volume, handle):
        task = Task(self._context, self._finalize_nccl, tag=self._tag)
        task.add_future_map(handle)
        task.execute(Rect([volume]))
