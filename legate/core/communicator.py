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

from abc import ABC, abstractmethod

from .launcher import TaskLauncher as Task
from .legion import Rect


class Communicator(ABC):
    def __init__(self):
        self._comms = {}

    def get_communicator(self, volume):
        if volume not in self._comms:
            self._comms[volume] = self._initialize(volume)
        return self._comms[volume]

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
        super(NCCLCommunicator, self).__init__()
        library = runtime.core_library

        self._runtime = runtime
        self._context = runtime.core_context
        self._init_nccl_id = library.LEGATE_CORE_INIT_NCCL_ID_TASK_ID
        self._init_nccl = library.LEGATE_CORE_INIT_NCCL_TASK_ID
        self._finalize = library.LEGATE_CORE_FINALIZE_NCCL_TASK_ID
        self._tag = library.LEGATE_GPU_VARIANT

    def _initialize(self, volume):
        # This doesn't need to run on a GPu, but will use it anyway
        task = Task(self._context, self._init_nccl_id, tag=self._tag)
        nccl_id = task.execute_single()

        task = Task(self._context, self._init_nccl, tag=self._tag)
        task.add_future(nccl_id)
        handle = task.execute(Rect([volume]))
        self._runtime.issue_execution_fence()
        return handle

    def _finalize(self, volume, handle):
        task = Task(self._context, self._finalize, tag=self._tag)
        task.add_future_map(handle)
        task.execute(Rect([handle]))
