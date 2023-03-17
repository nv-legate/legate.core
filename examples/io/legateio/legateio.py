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

import pyarrow as pa

import legate.core.types as ty
from legate.core import Array, Store

from .library import user_context as context, user_lib

_DTYPES = {
    "int": ty.int64,
    "float": ty.float64,
}


class LegateIOOpCode(IntEnum):
    READ = user_lib.cffi.READ
    WRITE = user_lib.cffi.WRITE


class Column:
    def __init__(self, store: Store, dtype: pa.DataType) -> None:
        self._store = store
        self._dtype = dtype

    @property
    def __legate_data_interface__(self) -> dict:
        array = Array(self._dtype, [None, self._store])
        field = pa.field("Legate IO Array", self._dtype, nullable=False)
        return {
            "version": 1,
            "data": {field: array},
        }


def read(filename: str, typename: str) -> Store:
    dtype = _DTYPES[typename]
    output = context.create_store(dtype, shape=None)
    task = context.create_auto_task(LegateIOOpCode.READ)

    task.add_output(output)
    task.add_scalar_arg(filename, ty.string)
    task.execute()

    return Column(output, dtype)
