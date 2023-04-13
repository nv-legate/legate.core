#!/usr/bin/env python3

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

import struct
from enum import IntEnum
from typing import Any

import numpy as np

import legate.core.types as types
from legate.core import Rect, Store, get_legate_runtime

from .library import user_context, user_lib


class HelloOpCode(IntEnum):
    HELLO_WORLD = user_lib.cffi.HELLO_WORLD
    SUM = user_lib.cffi.SUM
    SQUARE = user_lib.cffi.SQUARE
    IOTA = user_lib.cffi.IOTA


def print_hello(message: str) -> None:
    """Create a Legate task launch to print a message

    Args:
        message (str): The message to print
    """
    task = user_context.create_auto_task(HelloOpCode.HELLO_WORLD)
    task.add_scalar_arg(message, types.string)
    task.execute()


def print_hellos(message: str, n: int) -> None:
    """Create a Legate task launch to print a message n times,
       using n replicas of the task

    Args:
        message (str): The message to print
        n (int): The number of times to print
    """
    launch_domain = Rect(lo=[0], hi=[n])
    task = user_context.create_manual_task(
        HelloOpCode.HELLO_WORLD, launch_domain=launch_domain
    )
    task.add_scalar_arg(message, types.string)
    task.execute()


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


def to_scalar(input: Store) -> float:
    """Extracts a Python scalar value from a Legate store
       encapsulating a single scalar

    Args:
        input (Store): The Legate store encapsulating a scalar

    Returns:
        float: A Python scalar
    """
    # This operation blocks until the data in the Store
    # is available and correct
    buf = input.storage.get_buffer(np.float32().itemsize)
    result = np.frombuffer(buf, dtype=np.float32, count=1)
    return float(result[0])


def zero() -> Store:
    """Creates a Legate store representing a single zero scalar

    Returns:
        Store: A Legate store representing a scalar zero
    """
    data = bytearray(4)
    buf = struct.pack(f"{len(data)}s", data)
    future = get_legate_runtime().create_future(buf, len(buf))
    return user_context.create_store(
        types.float32,
        shape=(1,),
        storage=future,
        optimize_scalar=True,
    )


def iota(size: int) -> Store:
    """Enqueues a task that will generate a 1-D array
       1,2,...size.

    Args:
        size (int): The number of elements to generate

    Returns:
        Store: The Legate store that will hold the iota values
    """
    output = user_context.create_store(
        types.float32,
        shape=(size,),
        optimize_scalar=True,
    )
    task = user_context.create_auto_task(
        HelloOpCode.IOTA,
    )
    task.add_output(output)
    task.execute()
    return output


def sum(input: Any) -> Store:
    """Sums a 1-D array into a single scalar

    Args:
        input (Any): A Legate store or any object implementing
                     the Legate data interface.

    Returns:
        Store: A Legate store encapsulating the array sum
    """
    input_store = _get_legate_store(input)

    task = user_context.create_auto_task(HelloOpCode.SUM)

    # zero-initialize the output for the summation
    output = zero()

    task.add_input(input_store)
    task.add_reduction(output, types.ReductionOp.ADD)
    task.execute()
    return output


def square(input: Any) -> Store:
    """Computes the elementwise square of a 1-D array

    Args:
        input (Any): A Legate store or any object implementing
                     the Legate data interface.

    Returns:
        Store: A Legate store encapsulating a 1-D array
               holding the elementwise square values
    """
    input_store = _get_legate_store(input)

    output = user_context.create_store(
        types.float32, shape=input_store.shape, optimize_scalar=True
    )
    task = user_context.create_auto_task(HelloOpCode.SQUARE)

    task.add_input(input_store)
    task.add_output(output)
    task.add_alignment(input_store, output)
    task.execute()

    return output
