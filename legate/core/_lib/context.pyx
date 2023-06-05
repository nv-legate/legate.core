# Copyright 2023 NVIDIA Corporation
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

import cython
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "core/legate_c.h" nogil:
    ctypedef enum legate_core_variant_t:
        pass

cdef extern from "core/task/task_info.h" namespace "legate" nogil:
    cdef cppclass TaskInfo:
        bool has_variant(int)
        string name()

cdef extern from "core/runtime/context.h" namespace "legate" nogil:
    cdef cppclass LibraryContext:
        unsigned int get_task_id(long long)
        unsigned int get_mapper_id()
        int get_reduction_op_id(long long)
        unsigned int get_projection_id(long long)
        unsigned int get_sharding_id(long long)
        TaskInfo* find_task(long long)

cdef extern from "core/runtime/runtime.h" namespace "legate" nogil:
    cdef cppclass Runtime:
        @staticmethod
        Runtime* get_runtime()
        LibraryContext* find_library(string, bool)


cdef class CppTaskInfo:
    cdef const TaskInfo* _task_info

    @staticmethod
    cdef CppTaskInfo from_ptr(const TaskInfo* p_task_info):
        cdef CppTaskInfo result = CppTaskInfo.__new__(CppTaskInfo)
        result._task_info = p_task_info
        return result

    @property
    def valid(self) -> bool:
        return self._task_info != NULL

    @property
    def name(self) -> str:
        return self._task_info.name()

    def has_variant(self, int variant_id) -> bool:
        return self._task_info.has_variant(
            cython.cast(legate_core_variant_t, variant_id)
        )


cdef class Context:
    cdef LibraryContext* _context

    def __cinit__(self, str library_name, bool can_fail=False):
        self._context = Runtime.get_runtime().find_library(library_name.encode(), can_fail)

    def get_task_id(self, long long local_task_id) -> int:
        return self._context.get_task_id(local_task_id)

    def get_mapper_id(self) -> int:
        return self._context.get_mapper_id()

    def get_reduction_op_id(self, long long local_redop_id) -> int:
        return self._context.get_reduction_op_id(local_redop_id)

    def get_projection_id(self, long long local_proj_id) -> int:
        return self._context.get_projection_id(local_proj_id)

    def get_sharding_id(self, long long local_shard_id) -> int:
        return self._context.get_sharding_id(local_shard_id)

    def find_task(self, long long local_task_id) -> CppTaskInfo:
        return CppTaskInfo.from_ptr(self._context.find_task(local_task_id))
