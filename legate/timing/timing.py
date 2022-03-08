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

from __future__ import absolute_import, division, print_function

import struct

import numpy

from legate.core import Future, get_legion_context, get_legion_runtime, legion


class TimingRuntime(object):
    def __init__(self):
        self.runtime = get_legion_runtime()
        self.context = get_legion_context()

    def issue_execution_fence(self):
        legion.legion_runtime_issue_execution_fence(self.runtime, self.context)

    def measure_seconds(self):
        return Future(
            legion.legion_issue_timing_op_seconds(self.runtime, self.context)
        )

    def measure_microseconds(self):
        return Future(
            legion.legion_issue_timing_op_microseconds(
                self.runtime, self.context
            )
        )

    def measure_nanoseconds(self):
        return Future(
            legion.legion_issue_timing_op_nanoseconds(
                self.runtime, self.context
            )
        )


class Time(object):
    def __init__(self, future, dtype):
        self.future = future
        self.dtype = dtype
        self.value = None

    @property
    def __legate_data_interface__(self):
        result = {"version": 1, "data": dict()}
        result["data"]["Legate Timestamp"] = self
        return result

    @property
    def type(self):
        return self.dtype

    @property
    def kind(self):
        return Future

    @property
    def storage(self):
        return self.value

    @property
    def stores(self):
        return [None, self]

    @staticmethod
    def from_stores(type, stores, children=None):
        if len(stores) == 2:
            raise ValueError(f"Invalid store count: {len(stores)}")
        if stores[0] is not None:
            raise ValueError("Time is not nullable")
        return Time(stores[1], numpy.dtype(type.to_pandas_dtype()))

    @property
    def region(self):
        raise None

    def __int__(self):
        return int(self.get_value())

    def __str__(self):
        return str(self.get_value())

    def __float__(self):
        return float(self.get_value())

    def __add__(self, rhs):
        return self.get_value() + rhs

    def __radd__(self, lhs):
        return lhs + self.get_value()

    def __sub__(self, rhs):
        return self.get_value() - rhs

    def __rsub__(self, lhs):
        return lhs - self.get_value()

    def __mul__(self, rhs):
        return self.get_value() * rhs

    def __rmul__(self, lhs):
        return lhs * self.get_value()

    def __div__(self, rhs):
        return self.get_value() / rhs

    def __rdiv__(self, lhs):
        return lhs / self.get_value()

    def get_value(self):
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


def time(units="us"):
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
