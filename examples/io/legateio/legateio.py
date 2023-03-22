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

import pyarrow as pa

import legate.core.types as ty
from legate.core import Array, Rect, Store, get_legate_runtime

from .library import user_context as context, user_lib


class LegateIOOpCode(IntEnum):
    READ_EVEN_TILES = user_lib.cffi.READ_EVEN_TILES
    READ_FILE = user_lib.cffi.READ_FILE
    READ_UNEVEN_TILES = user_lib.cffi.READ_UNEVEN_TILES
    WRITE_EVEN_TILES = user_lib.cffi.WRITE_EVEN_TILES
    WRITE_FILE = user_lib.cffi.WRITE_FILE
    WRITE_UNEVEN_TILES = user_lib.cffi.WRITE_UNEVEN_TILES


_CODES_TO_DTYPES = dict(
    (context.type_system[t].code, t)
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
    def __init__(self, store: Store, dtype: pa.lib.DataType) -> None:
        self._store = store
        self._dtype = dtype

    @staticmethod
    def from_legate_data_interface(data: dict[str, Any]) -> "IOArray":
        assert data["version"] == 1
        field = next(iter(data["data"]))
        stores = data["data"][field].stores()

        # We can only import non-nullable containers
        assert len(stores) == 2 and stores[0] is None
        _, store = stores
        return IOArray(store, field.type)

    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        array = Array(self._dtype, [None, self._store])
        field = pa.field("Legate IO Array", self._dtype, nullable=False)
        return {
            "version": 1,
            "data": {field: array},
        }

    def to_file(self, filename: str) -> None:
        task = context.create_auto_task(LegateIOOpCode.WRITE_FILE)

        task.add_input(self._store)
        task.add_scalar_arg(filename, ty.string)
        task.add_broadcast(self._store)
        task.execute()

    def to_uneven_tiles(self, dirname: str) -> None:
        os.mkdir(dirname)

        task = context.create_auto_task(LegateIOOpCode.WRITE_UNEVEN_TILES)

        task.add_input(self._store)
        task.add_scalar_arg(dirname, ty.string)
        task.execute()

    def to_even_tiles(self, dirname: str, tile_shape: tuple[int, ...]) -> None:
        os.mkdir(dirname)

        store_partition = self._store.partition_by_tiling(tile_shape)
        launch_shape = store_partition.partition.color_shape

        task = context.create_manual_task(
            LegateIOOpCode.WRITE_EVEN_TILES,
            launch_domain=Rect(launch_shape),
        )

        task.add_input(store_partition)
        task.add_scalar_arg(dirname, ty.string)
        task.add_scalar_arg(self._store.shape, (ty.int32,))
        task.add_scalar_arg(tile_shape, (ty.int32,))
        task.execute()


def read_file(filename: str, dtype: pa.lib.DataType) -> IOArray:
    output = context.create_store(dtype, shape=None, ndim=1)
    task = context.create_auto_task(LegateIOOpCode.READ_FILE)

    task.add_output(output)
    task.add_scalar_arg(filename, ty.string)
    task.execute()

    return IOArray(output, dtype)


def read_file_parallel(
    filename: str, dtype: pa.lib.DataType, parallelism: Optional[int] = None
) -> IOArray:
    if parallelism is None:
        parallelism = get_legate_runtime().num_procs

    output = context.create_store(dtype, shape=None, ndim=1)
    task = context.create_manual_task(
        LegateIOOpCode.READ_FILE,
        launch_domain=Rect([parallelism]),
    )

    task.add_output(output)
    task.add_scalar_arg(filename, ty.string)
    task.execute()

    return IOArray(output, dtype)


def _read_header_uneven(dirname: str) -> tuple[int, ...]:
    with open(os.path.join(dirname, ".header"), "rb") as f:
        data = f.read()
        (
            code,
            dim,
        ) = struct.unpack("ii", data[:8])
        return code, struct.unpack(f"{dim}q", data[8:])


def read_uneven_tiles(dirname: str) -> IOArray:
    code, launch_shape = _read_header_uneven(dirname)
    dtype = _CODES_TO_DTYPES[code]
    output = context.create_store(dtype, shape=None, ndim=len(launch_shape))
    task = context.create_manual_task(
        LegateIOOpCode.READ_UNEVEN_TILES,
        launch_domain=Rect(launch_shape),
    )

    task.add_output(output)
    task.add_scalar_arg(dirname, ty.string)
    task.execute()

    return IOArray(output, dtype)


def _read_header_even(dirname: str) -> tuple[int, ...]:
    with open(os.path.join(dirname, ".header"), "rb") as f:
        data = f.read()
        (
            code,
            dim,
        ) = struct.unpack("ii", data[:8])
        data = data[8:]
        shape = struct.unpack(f"{dim}i", data[: 4 * dim])
        tile_shape = struct.unpack(f"{dim}i", data[4 * dim :])
        return code, shape, tile_shape


def read_even_tiles(dirname: str) -> IOArray:
    code, shape, tile_shape = _read_header_even(dirname)
    dtype = _CODES_TO_DTYPES[code]
    output = context.create_store(dtype, shape=shape)
    output_partition = output.partition_by_tiling(tile_shape)
    launch_shape = output_partition.partition.color_shape
    task = context.create_manual_task(
        LegateIOOpCode.READ_EVEN_TILES,
        launch_domain=Rect(launch_shape),
    )

    task.add_output(output_partition)
    task.add_scalar_arg(dirname, ty.string)
    task.execute()

    return IOArray(output, dtype)
