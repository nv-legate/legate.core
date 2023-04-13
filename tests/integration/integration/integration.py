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

from enum import IntEnum
from typing import Any, Tuple

# import legate.core.types as ty
from legate.core import Rect, Store  # , get_legate_runtime

from .library import user_context as context, user_lib  # type: ignore


class OpCode(IntEnum):
    COLLECTIVE = user_lib.cffi.COLLECTIVE


def _get_legate_store(input: Any) -> Store:
    """Extracts a Legate store from any object
       implementing the legete data interface

    Args:
        input (Any): The input object

    Returns:
        Store: The extracted Legate store
    """
    if isinstance(input, Store):
        return input
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store


# def create_int64_store(shape:Tuple[Any,...])->Store:
#    store =  context.create_store(ty.int64, shape=shape)
#    #call empty function on the store to make Legion think
#    # that we initialized the store
#    task = context.create_manual_task(
#        OpCode.COLLECTIVE,
#        launch_domain=Rect(launch_shape),
#    )
#    task.add_output(store)
#    # we use errors from the mapper to check correct behavior
#    task.throws_exception(RuntimeError)
#    task.execute()
#    return store


def _broadcast(store: Store, shape: Tuple[Any, ...]) -> Store:
    result = store
    diff = len(shape) - result.ndim
    for dim in range(diff):
        result = result.promote(dim, shape[dim])

    for dim in range(len(shape)):
        if result.shape[dim] != shape[dim]:
            if result.shape[dim] != 1:
                raise ValueError(
                    f"Shape did not match along dimension {dim} "
                    "and the value is not equal to 1"
                )
            result = result.project(dim, 0).promote(dim, shape[dim])
    return result


def collective_test(
    store: Store, shape: Tuple[Any, ...], tile_shape: Tuple[Any, ...]
) -> None:
    assert store.ndim == 2
    if store.shape != shape:
        store = _broadcast(store, shape)

    store_partition = store.partition_by_tiling(tile_shape)
    launch_shape = store_partition.partition.color_shape

    task = context.create_manual_task(
        OpCode.COLLECTIVE,
        launch_domain=Rect(launch_shape),
    )
    task.add_input(store_partition)
    # we use errors from the mapper to check correct behavior
    task.execute()
