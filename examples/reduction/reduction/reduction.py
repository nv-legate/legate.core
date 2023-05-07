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

import cunumeric as np

import legate.core.types as ty
from legate.core import Store

from .library import user_context as context, user_lib


class OpCode(IntEnum):
    BINCOUNT = user_lib.cffi.BINCOUNT
    CATEGORIZE = user_lib.cffi.CATEGORIZE
    HISTOGRAM = user_lib.cffi.HISTOGRAM
    MATMUL = user_lib.cffi.MATMUL
    MUL = user_lib.cffi.MUL
    SUM_OVER_AXIS = user_lib.cffi.SUM_OVER_AXIS
    UNIQUE = user_lib.cffi.UNIQUE


def _sanitize_axis(axis: int, ndim: int) -> int:
    sanitized = axis
    if sanitized < 0:
        sanitized += ndim
    if sanitized < 0 or sanitized >= ndim:
        raise ValueError(f"Invalid axis {axis} for a {ndim}-D store")
    return sanitized


def sum_over_axis(input: Store, axis: int) -> Store:
    """
    Sum values along the chosen axis

    Parameters
    ----------
    input : Store
        Input to sum
    axis : int
        Axis along which the summation should be done

    Returns
    -------
    Store
        Summation result
    """
    sanitized = _sanitize_axis(axis, input.ndim)

    # Compute the output shape by removing the axis being summed over
    res_shape = tuple(
        ext for dim, ext in enumerate(input.shape) if dim != sanitized
    )
    result = context.create_store(input.type, res_shape)
    np.asarray(result).fill(0)

    # Broadcast the output along the contracting dimension
    promoted = result.promote(axis, input.shape[axis])

    assert promoted.shape == input.shape

    task = context.create_auto_task(OpCode.SUM_OVER_AXIS)
    task.add_input(input)
    task.add_reduction(promoted, ty.ReductionOp.ADD)
    task.add_alignment(input, promoted)

    task.execute()

    return result


def multiply(rhs1: Store, rhs2: Store) -> Store:
    if rhs1.type != rhs2.type or rhs1.shape != rhs2.shape:
        raise ValueError("Stores to add must have the same type and shape")

    result = context.create_store(rhs1.type, rhs1.shape)

    task = context.create_auto_task(OpCode.MUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_output(result)
    task.add_alignment(result, rhs1)
    task.add_alignment(result, rhs2)

    task.execute()

    return result


def matmul(rhs1: Store, rhs2: Store) -> Store:
    """
    Performs matrix multiplication

    Parameters
    ----------
    rhs1, rhs2 : Store
        Matrices to multiply

    Returns
    -------
    Store
        Multiplication result
    """
    if rhs1.ndim != 2 or rhs2.ndim != 2:
        raise ValueError("Stores must be 2D")
    if rhs1.type != rhs2.type:
        raise ValueError("Stores must have the same type")
    if rhs1.shape[1] != rhs2.shape[0]:
        raise ValueError(
            "Can't do matrix mulplication between arrays of "
            f"shapes {rhs1.shape} and {rhs1.shape}"
        )

    m = rhs1.shape[0]
    k = rhs1.shape[1]
    n = rhs2.shape[1]

    # Multiplying an (m, k) matrix with a (k, n) matrix gives
    # an (m, n) matrix
    result = context.create_store(rhs1.type, (m, n))
    np.asarray(result).fill(0)

    # Each store gets a fake dimension that it doesn't have
    rhs1 = rhs1.promote(2, n)
    rhs2 = rhs2.promote(0, m)
    lhs = result.promote(1, k)

    assert lhs.shape == rhs1.shape
    assert lhs.shape == rhs2.shape

    task = context.create_auto_task(OpCode.MATMUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_reduction(lhs, ty.ReductionOp.ADD)
    task.add_alignment(lhs, rhs1)
    task.add_alignment(lhs, rhs2)

    task.execute()

    return result


def bincount(input: Store, num_bins: int) -> Store:
    """
    Counts the occurrences of each bin index

    Parameters
    ----------
    input : Store
        Input to bin-count
    num_bins : int
        Number of bins

    Returns
    -------
    Store
        Counting result
    """
    result = context.create_store(ty.uint64, (num_bins,))
    np.asarray(result).fill(0)

    task = context.create_auto_task(OpCode.BINCOUNT)
    task.add_input(input)
    # Broadcast the result store. This commands the Legate runtime to give
    # the entire store to every task instantiated by this task descriptor
    task.add_broadcast(result)
    # Declares that the tasks will do reductions to the result store and
    # that outputs from the tasks should be combined by addition
    task.add_reduction(result, ty.ReductionOp.ADD)

    task.execute()

    return result


def categorize(input: Store, bins: Store) -> Store:
    result = context.create_store(ty.uint64, input.shape)

    task = context.create_auto_task(OpCode.CATEGORIZE)
    task.add_input(input)
    task.add_input(bins)
    task.add_output(result)

    # Broadcast the store that contains bin edges. Each task will get a copy
    # of the entire bin edges
    task.add_broadcast(bins)

    task.execute()

    return result


def histogram(input: Store, bins: Store) -> Store:
    """
    Constructs a histogram for the given bins

    Parameters
    ----------
    input : Store
        Input
    bins : int
        Bin edges

    Returns
    -------
    Store
        Histogram
    """
    num_bins = bins.shape[0] - 1
    result = context.create_store(ty.uint64, (num_bins,))
    np.asarray(result).fill(0)

    task = context.create_auto_task(OpCode.HISTOGRAM)
    task.add_input(input)
    task.add_input(bins)
    task.add_reduction(result, ty.ReductionOp.ADD)

    # Broadcast both the result store and the one that contains bin edges.
    task.add_broadcast(bins)
    task.add_broadcast(result)

    task.execute()

    return result


def unique(input: Store, radix: int = 4) -> Store:
    """
    Finds unique elements in the input and returns them in a store

    Parameters
    ----------
    input : Store
        Input

    Returns
    -------
    Store
        Result that contains only the unique elements of the input
    """

    if input.ndim > 1:
        raise ValueError("`unique` accepts only 1D stores")

    if input.type in (
        ty.float16,
        ty.float32,
        ty.float64,
        ty.complex64,
        ty.complex128,
    ):
        raise ValueError(
            "`unique` doesn't support floating point or complex numbers"
        )

    # Create an unbound store to collect local results
    result = context.create_store(input.type, shape=None, ndim=1)

    task = context.create_auto_task(OpCode.UNIQUE)
    task.add_input(input)
    task.add_output(result)

    task.execute()

    # Perform global reduction using a reduction tree
    return context.tree_reduce(OpCode.UNIQUE, result, radix=radix)
