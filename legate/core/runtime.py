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

from .legate import get_legion_runtime, legate_add_library
from .legion import legion


class ResourceConfig(object):
    __slots__ = [
        "max_tasks",
        "max_mappers",
        "max_reduction_ops",
        "max_projections",
        "max_shardings",
    ]

    def __init__(self):
        self.max_tasks = 1_000_000
        self.max_mappers = 1
        self.max_reduction_ops = 0
        self.max_projections = 0
        self.max_shardings = 0


class Context(object):
    def __init__(self, runtime, library):
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
        self._name = library.name

        config = library.resource_config

        name = self._name.encode("utf-8")
        legion_runtime = get_legion_runtime()

        def _maybe_generate_ids(api, max_counts):
            if max_counts > 0:
                return api(legion_runtime, name, max_counts)
            else:
                return None

        self._first_task_id = _maybe_generate_ids(
            legion.legion_runtime_generate_library_task_ids,
            config.max_tasks,
        )
        self._first_mapper_id = _maybe_generate_ids(
            legion.legion_runtime_generate_library_mapper_ids,
            config.max_mappers,
        )
        self._first_redop_id = _maybe_generate_ids(
            legion.legion_runtime_generate_library_reduction_ids,
            config.max_reduction_ops,
        )
        self._first_proj_id = _maybe_generate_ids(
            legion.legion_runtime_generate_library_projection_ids,
            config.max_projections,
        )
        self._first_shard_id = _maybe_generate_ids(
            legion.legion_runtime_generate_library_sharding_ids,
            config.max_shardings,
        )


class Runtime(object):
    def __init__(self):
        """
        This is a class that implements the Legate runtime.
        The Runtime object provides high-level APIs for Legate libraries
        to use services in the Legion runtime. The Runtime centralizes
        resource management for all the libraries so that they can
        focuse on implementing their domain logic.
        """
        self._contexts = {}

    def register_library(self, library):
        if library.name in self._contexts:
            raise RuntimeError(f"library {library.name} already exists!")
        legate_add_library(library)
        context = Context(self, library)
        self._contexts[library.name] = context
        return context


_runtime = Runtime()


def get_legate_runtime():
    return _runtime
