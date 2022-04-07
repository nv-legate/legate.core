# Copyright 2021-2022 NVIDIA Corporation
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
from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any, Optional, Type, Union

import numpy

from ..core import Future, get_legion_context, get_legion_runtime, legion

if TYPE_CHECKING:
    import pyarrow


class TimingRuntime(object):
    def __init__(self) -> None:
        self.runtime = get_legion_runtime()
        self.context = get_legion_context()

    def issue_execution_fence(self) -> None:
        legion.legion_runtime_issue_execution_fence(self.runtime, self.context)

    def measure_seconds(self) -> Future:
        return Future(
            legion.legion_issue_timing_op_seconds(self.runtime, self.context)
        )

    def measure_microseconds(self) -> Future:
        return Future(
            legion.legion_issue_timing_op_microseconds(
                self.runtime, self.context
            )
        )

    def measure_nanoseconds(self) -> Future:
        return Future(
            legion.legion_issue_timing_op_nanoseconds(
                self.runtime, self.context
            )
        )


class Time(object):
    def __init__(self, future: Future, dtype: Any) -> None:
        self.future = future
        self.dtype = dtype
        self.value: Union[int, float, None] = None

    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        result: dict[str, Any] = {"version": 1, "data": dict()}
        result["data"]["Legate Timestamp"] = self
        return result

    @property
    def type(self) -> Any:
        return self.dtype

    @property
    def kind(self) -> Type[Future]:
        return Future

    @property
    def storage(self) -> Union[int, float, None]:
        return self.value

    @property
    def stores(self) -> list[Any]:
        return [None, self]

    @staticmethod
    def from_stores(
        type: pyarrow.DataType,
        stores: list[Any],
        children: Optional[Any] = None,
    ) -> Time:
        if len(stores) == 2:
            raise ValueError(f"Invalid store count: {len(stores)}")
        if stores[0] is not None:
            raise ValueError("Time is not nullable")
        return Time(stores[1], numpy.dtype(type.to_pandas_dtype()))

    @property
    def region(self) -> None:
        raise RuntimeError()

    def __int__(self) -> int:
        return int(self.get_value())

    def __str__(self) -> str:
        return str(self.get_value())

    def __float__(self) -> float:
        return float(self.get_value())

    def __add__(self, rhs: Union[int, float]) -> Union[int, float]:
        return self.get_value() + rhs

    def __radd__(self, lhs: Union[int, float]) -> Union[int, float]:
        return lhs + self.get_value()

    def __sub__(self, rhs: Union[int, float]) -> Union[int, float]:
        return self.get_value() - rhs

    def __rsub__(self, lhs: Union[int, float]) -> Union[int, float]:
        return lhs - self.get_value()

    def __mul__(self, rhs: Union[int, float]) -> Union[int, float]:
        return self.get_value() * rhs

    def __rmul__(self, lhs: Union[int, float]) -> Union[int, float]:
        return lhs * self.get_value()

    def __div__(self, rhs: Union[int, float]) -> float:
        return self.get_value() / rhs

    def __rdiv__(self, lhs: Union[int, float]) -> float:
        return lhs / self.get_value()

    def get_value(self) -> Union[int, float]:
        if self.value is None:
            if self.dtype == numpy.int64:
                self.value = struct.unpack_from(
                    "q", self.future.get_buffer(8)
                )[0]
            else:
                assert self.dtype == numpy.float64
                self.value = struct.unpack_from(
                    "d", self.future.get_buffer(8)
                )[0]
        return self.value


_timing = TimingRuntime()


def time(units: str = "us") -> Time:
    # Issue a Legion execution fence and then perform a timing operation
    # immediately after it
    _timing.issue_execution_fence()
    if units == "s":
        return Time(_timing.measure_seconds(), numpy.float64)
    elif units == "us":
        return Time(_timing.measure_microseconds(), numpy.int64)
    elif units == "ns":
        return Time(_timing.measure_nanoseconds(), numpy.int64)
    else:
        raise ValueError('time units must be one of "s", "us", or "ns"')
