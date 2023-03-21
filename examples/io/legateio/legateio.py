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
from typing import Any, Optional

import pyarrow as pa

import legate.core.types as ty
from legate.core import Array, Rect, Store, get_legate_runtime

from .library import user_context as context, user_lib


class LegateIOOpCode(IntEnum):
    READ_FILE = user_lib.cffi.READ_FILE
    WRITE_FILE = user_lib.cffi.WRITE_FILE


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
