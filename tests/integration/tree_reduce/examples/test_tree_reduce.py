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

import pytest
from tree_reduce import user_context, user_lib

import legate.core.types as ty
from legate.core import Rect


def test_tree_reduce_normal():
    num_tasks = user_lib.cffi.NUM_NORMAL_PRODUCER
    tile_size = user_lib.cffi.TILE_SIZE
    task = user_context.create_manual_task(
        user_lib.shared_object.PRODUCE_NORMAL, Rect([num_tasks])
    )
    store = user_context.create_store(ty.int64, shape=(num_tasks * tile_size,))
    part = store.partition_by_tiling((tile_size,))
    task.add_output(part)
    task.execute()

    result = user_context.tree_reduce(
        user_lib.shared_object.REDUCE_NORMAL, store, radix=4
    )
    # The result should be a normal store
    assert not result.unbound


def test_tree_reduce_unbound():
    num_tasks = 4
    task = user_context.create_manual_task(
        user_lib.shared_object.PRODUCE_UNBOUND, Rect([num_tasks])
    )
    store = user_context.create_store(ty.int64, ndim=1)
    task.add_output(store)
    task.execute()

    result = user_context.tree_reduce(
        user_lib.shared_object.REDUCE_UNBOUND, store, radix=num_tasks
    )
    # The result should be a normal store
    assert not result.unbound


def test_tree_single_proc():
    task = user_context.create_manual_task(
        user_lib.shared_object.PRODUCE_UNBOUND, Rect([1])
    )
    store = user_context.create_store(ty.int64, ndim=1)
    task.add_output(store)
    task.execute()

    result = user_context.tree_reduce(
        user_lib.shared_object.REDUCE_UNBOUND, store, radix=4
    )
    # The result should be a normal store
    assert not result.unbound


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
