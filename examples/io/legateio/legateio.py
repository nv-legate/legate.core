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

import os
import struct
from enum import IntEnum
from typing import Any, Optional

import legate.core.types as ty
from legate.core import Array, Field, Rect, Store, get_legate_runtime

from .library import user_context as context, user_lib


class LegateIOOpCode(IntEnum):
    READ_EVEN_TILES = user_lib.cffi.READ_EVEN_TILES
    READ_FILE = user_lib.cffi.READ_FILE
    READ_UNEVEN_TILES = user_lib.cffi.READ_UNEVEN_TILES
    WRITE_EVEN_TILES = user_lib.cffi.WRITE_EVEN_TILES
    WRITE_FILE = user_lib.cffi.WRITE_FILE
    WRITE_UNEVEN_TILES = user_lib.cffi.WRITE_UNEVEN_TILES


# Mapping between type codes and type objects
# We use this mapping to create an array based on the type information
# from a dataset's header file.
_CODES_TO_DTYPES = dict(
    (t.code, t)
    for t in (
        ty.bool_,
        ty.int8,
        ty.int16,
        ty.int32,
        ty.int64,
        ty.uint8,
        ty.uint16,
        ty.uint32,
        ty.uint64,
        ty.float16,
        ty.float32,
        ty.float64,
        ty.complex64,
        ty.complex128,
    )
)


class IOArray:
    """
    A simple array implementation used in the tutorial.

    This array can be passed to any Legate library call that supports Legate
    data interface.
    """

    def __init__(self, store: Store, dtype: ty.Dtype) -> None:
        self._store = store
        self._dtype = dtype

    @staticmethod
    def from_legate_data_interface(data: dict[str, Any]) -> "IOArray":
        """
        Constructs an IOArray from a Legate data interface object.

        The Legate data interface object contains its data under the
        ``data`` field, which is a map from fields to arrays. Each array
        is backed by one or more stores, and their layout follows Apache
        arrow's data layout. For example, a nullable fixed type array
        would be backed by two stores, one for the bitmask and one for
        the actual array data.
        """

        assert data["version"] == 1
        # For now, we assume that there's only one field in the container
        field: Field = next(iter(data["data"]))
        stores = data["data"][field].stores()

        # We only support non-nullable arrays
        assert len(stores) == 2 and stores[0] is None
        _, store = stores
        return IOArray(store, field.type)

    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        """
        Constructs a Legate data interface object from this IOArray.
        """

        # Every IOArray would be mapped to a non-nullable Legate array
        # of a fixed type
        array = Array(self._dtype, [None, self._store])

        # Create a field metadata to populate the data field
        field = Field("Legate IO Array", self._dtype)

        return {
            "version": 1,
            "data": {field: array},
        }

    def to_file(self, filename: str) -> None:
        """
        Dumps the IOArray to a single file.

        Only a single task will be launched by this call.
        """
        task = context.create_auto_task(LegateIOOpCode.WRITE_FILE)

        task.add_input(self._store)
        task.add_scalar_arg(filename, ty.string)
        # Request a broadcasting for the input. Since this is the only store
        # argument to the task, Legate will launch a single task from this
        # task descriptor.
        task.add_broadcast(self._store)
        task.execute()

    def to_uneven_tiles(self, path: str) -> None:
        """
        Dumps the IOArray into (potentially) uneven tiles

        Parameters
        ----------
        path : str
            Path to the dataset
        """
        os.mkdir(path)

        task = context.create_auto_task(LegateIOOpCode.WRITE_UNEVEN_TILES)

        task.add_input(self._store)
        task.add_scalar_arg(path, ty.string)
        task.execute()

    def to_even_tiles(self, path: str, tile_shape: tuple[int, ...]) -> None:
        """
        Dumps the IOArray into even tiles

        Parameters
        ----------
        path : str
            Path to the dataset
        tile_shape : tuple[int]
            Shape of the tiles
        """
        os.mkdir(path)

        # Partition the array into even tiles
        store_partition = self._store.partition_by_tiling(tile_shape)

        # Use the partition's color shape as the launch shape so there will be
        # one task for each tile
        launch_shape = store_partition.partition.color_shape

        task = context.create_manual_task(
            LegateIOOpCode.WRITE_EVEN_TILES,
            launch_domain=Rect(launch_shape),
        )

        task.add_input(store_partition)
        task.add_scalar_arg(path, ty.string)
        # Pass the input shape and the tile shape so the task can generate a
        # header file
        task.add_scalar_arg(self._store.shape, (ty.int32,))
        task.add_scalar_arg(tile_shape, (ty.int32,))
        task.execute()


def read_file(filename: str, dtype: ty.Dtype) -> IOArray:
    """
    Reads a file into an IOArray.

    Only a single task will be launched by this call.

    Parameters
    ----------
    filename : str
        File name
    dtype : DataType
        Data type. The task will fail with an error if it's inconsistent
        with the type information in the file

    Returns
    -------
    IOArray
        An array that contains data from the file
    """

    # Create a 1D unbound store by passing None to the shape
    output = context.create_store(dtype, shape=None, ndim=1)
    task = context.create_auto_task(LegateIOOpCode.READ_FILE)

    task.add_output(output)
    task.add_scalar_arg(filename, ty.string)

    # Legate as it stands today will launch a single task from this request.
    # Here's why: the READ_FILE task only has one store argument that is
    # unbound and thus has no shape information that Legate's auto-partitioner
    # would normally extract to determine the granularity of partition.
    # So, Legate simply falls back to a single task launch.
    #
    # This behavior can change in the future when the auto-partitioner
    # looks ahead and finds a downstream consumer that can be parallelized,
    # in which case this task would get parallelized in the aligned way.
    task.execute()

    return IOArray(output, dtype)


def read_file_parallel(
    filename: str, dtype: ty.Dtype, parallelism: Optional[int] = None
) -> IOArray:
    """
    Reads a file into an IOArray using multiple tasks.

    Parameters
    ----------
    filename : str
        File name
    dtype : DataType
        Data type. The task will fail with an error if it's inconsistent
        with the type information in the file
    parallelism: int, optional
        Degree of parallelism for reader tasks. If None, it will be set to
        the number of processors.


    Returns
    -------
    IOArray
        An array that contains data from the file
    """

    if parallelism is None:
        # the num_procs property returns the number of processors that
        # Legate will favor to launch tasks
        parallelism = get_legate_runtime().num_procs

    # Create an 1D unbound store
    #
    # The shape of this store is determined by outputs from the tasks.
    # For example, if there are four reader tasks that respectively read
    # 2, 3, 4, and 5 elements from the file, the store's shape would become
    # (14,) and be partitioned internally into the following ranges, all of
    # which are computed by the runtime:
    #
    #   [0, 2), [2, 5), [5, 9), [9, 14)
    #
    output = context.create_store(dtype, shape=None, ndim=1)

    # Create a manually parallelized task
    task = context.create_manual_task(
        LegateIOOpCode.READ_FILE,
        launch_domain=Rect([parallelism]),
    )

    task.add_output(output)
    task.add_scalar_arg(filename, ty.string)
    task.execute()

    return IOArray(output, dtype)


def _read_header_uneven(path: str) -> tuple[int, ...]:
    with open(os.path.join(path, ".header"), "rb") as f:
        data = f.read()
        (code, dim) = struct.unpack("ii", data[:8])
        return code, struct.unpack(f"{dim}q", data[8:])


def read_uneven_tiles(path: str) -> IOArray:
    """
    Reads a dataset of uneven tiles

    Parameters
    ----------
    path : str
        Path to the dataset

    Returns
    -------
    IOArray
        An array that contains data from the dataset
    """
    # Read the dataset's header to find the type code and the color shape of
    # the partition, which are laid out in the header in the following way:
    #
    #   +-----------+--------+----------+-----
    #   | type code | # dims | extent 0 | ...
    #   |   (4B)    |  (4B)  |   (8B)   |
    #   +-----------+--------+----------+-----
    #
    code, color_shape = _read_header_uneven(path)

    # Convert the type code into a type object
    dtype = _CODES_TO_DTYPES[code]

    # Create a multi-dimensional unbound store
    #
    # Like 1D unbound stores, the shape of this store is also determined by
    # outputs from the tasks. For example, if the store is 2D and there are
    # four tasks (0, 0), (0, 1), (1, 0), and (1, 1) that respectively produce
    # 2x3, 2x4, 3x3, and 3x4 outputs, the store's shape would be (5, 7) and
    # internally partitioned in the following way:
    #
    #           0  1  2  3  4  5  6
    #         +--------+------------+
    #       0 | (0, 0) |   (0, 1)   |
    #       1 |        |            |
    #         +--------+------------+
    #       2 |        |            |
    #       3 | (1, 0) |   (1, 1)   |
    #       4 |        |            |
    #         +--------+------------+
    #

    output = context.create_store(dtype, shape=None, ndim=len(color_shape))

    task = context.create_manual_task(
        LegateIOOpCode.READ_UNEVEN_TILES,
        launch_domain=Rect(color_shape),
    )

    task.add_output(output)
    task.add_scalar_arg(path, ty.string)
    task.execute()

    return IOArray(output, dtype)


def _read_header_even(path: str) -> tuple[int, ...]:
    with open(os.path.join(path, ".header"), "rb") as f:
        data = f.read()
        (code, dim) = struct.unpack("ii", data[:8])
        data = data[8:]
        shape = struct.unpack(f"{dim}i", data[: 4 * dim])
        tile_shape = struct.unpack(f"{dim}i", data[4 * dim :])
        return code, shape, tile_shape


def read_even_tiles(path: str) -> IOArray:
    """
    Reads a dataset of even tiles

    Parameters
    ----------
    path : str
        Path to the dataset

    Returns
    -------
    IOArray
        An array that contains data from the dataset
    """
    # Read the dataset's header to find the type code, the array's shape, and
    # the tile shape. The following shows the header's format:
    #
    #   +-----------+--------+----------+-----+----------+-----
    #   |           |        |  shape   |     |   tile   |
    #   | type code | # dims | extent 0 | ... | extent 0 | ...
    #   |   (4B)    |  (4B)  |   (4B)   |     |   (4B)   |
    #   +-----------+--------+----------+-----+----------+-----
    #
    code, shape, tile_shape = _read_header_even(path)

    # Convert the type code into a type object
    dtype = _CODES_TO_DTYPES[code]

    # Since the shape of the array is known, we can create a normal store,
    output = context.create_store(dtype, shape=shape)

    # and partition the array into evenly shaped tiles
    output_partition = output.partition_by_tiling(tile_shape)

    launch_shape = output_partition.partition.color_shape
    task = context.create_manual_task(
        LegateIOOpCode.READ_EVEN_TILES,
        launch_domain=Rect(launch_shape),
    )

    task.add_output(output_partition)
    task.add_scalar_arg(path, ty.string)
    task.execute()

    # Unlike AutoTask, manually parallelized tasks don't update the "key"
    # partition for their outputs.
    output.set_key_partition(output_partition.partition)

    return IOArray(output, dtype)
