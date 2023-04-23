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
from abc import abstractmethod
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np

from .. import legion
from .field import FieldID
from .geometry import Point
from .partition import IndexPartition
from .pending import _pending_deletions, _pending_unordered

if TYPE_CHECKING:
    from . import AffineTransform


def legate_task_preamble(
    runtime: legion.legion_runtime_t, context: legion.legion_context_t
) -> None:
    """
    This function sets up internal Legate state for a task in Python.
    In general, users only need to worry about calling this function
    at the beginning of sub-tasks on the Python side. The Legate
    Core will perform the necessary call to this function for the
    top-level task.
    """
    assert context not in _pending_unordered
    _pending_unordered[context] = list()


def legate_task_progress(
    runtime: legion.legion_runtime_t, context: legion.legion_context_t
) -> None:
    """
    This method will progress any internal Legate Core functionality
    that is running in the background. Legate clients do not need to
    call it, but can optionally do so to speed up collection.
    """

    from .future import Future, FutureMap
    from .operation import Detach
    from .region import OutputRegion, PhysicalRegion, Region
    from .space import FieldSpace, IndexSpace
    from .task import ArgumentMap

    # The context should always be in the set of pending deletions
    deletions = _pending_unordered[context]
    if deletions:
        for handle, type in deletions:
            if type is IndexSpace:
                legion.legion_index_space_destroy_unordered(
                    runtime, context, handle[0], True
                )
            elif type is IndexPartition:
                legion.legion_index_partition_destroy_unordered(
                    runtime, context, handle[0], True, handle[1]
                )
            elif type is FieldSpace:
                legion.legion_field_space_destroy_unordered(
                    runtime, context, handle[0], True
                )
                if handle[1] is not None:
                    legion.legion_field_allocator_destroy(handle[1])
            elif type is FieldID:
                legion.legion_field_allocator_free_field_unordered(
                    handle[0], handle[1], True
                )
            elif type is Region:
                legion.legion_logical_region_destroy_unordered(
                    runtime, context, handle, True
                )
            elif type is PhysicalRegion:
                handle.unmap(runtime, context, unordered=False)
            elif type is Detach:
                detach = handle[0]
                future = handle[1]
                assert future.handle is None
                future.handle = (
                    legion.legion_unordered_detach_external_resource(
                        runtime,
                        context,
                        detach.physical_region.handle,
                        detach.flush,
                        True,
                    )
                )
            else:
                raise TypeError(
                    "Internal legate type error on unordered operations"
                )
        deletions.clear()
    if _pending_deletions:
        for handle, type in _pending_deletions:
            if type is Future:
                legion.legion_future_destroy(handle)
            elif type is FutureMap:
                legion.legion_future_map_destroy(handle)
            elif type is PhysicalRegion:
                legion.legion_physical_region_destroy(handle)
            elif type is ArgumentMap:
                legion.legion_argument_map_destroy(handle)
            elif type is OutputRegion:
                legion.legion_output_requirement_destroy(handle)
            elif type is ExternalResources:
                legion.legion_external_resources_destroy(handle)
            else:
                raise TypeError(
                    "Internal legate type error on pending deletions"
                )
        _pending_deletions.clear()


def legate_task_postamble(
    runtime: legion.legion_runtime_t, context: legion.legion_context_t
) -> None:
    """
    This function cleans up internal Legate state for a task in Python.
    In general, users only need to worry about calling this function
    at the end of sub-tasks on the Python side. The Legate
    Core will perform the necessary call to this function for the
    top-level task.
    """
    legate_task_progress(runtime, context)
    del _pending_unordered[context]


# This is a decorator for wrapping the launch method on launchers
# to dispatch any unordered deletions while the task is live
def dispatch(func: Any) -> Any:
    def launch(
        launcher: Any,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> Any:
        # This context should always be in the dictionary
        legate_task_progress(runtime, context)
        return func(launcher, runtime, context, **kwargs)

    return launch


T = TypeVar("T")


class Dispatchable(Generic[T]):
    @abstractmethod
    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> T:
        ...


class Mappable(Protocol):
    def set_mapper_arg(self, data: Any, size: int) -> None:
        ...


# todo: (bev) use list[...] when feasible
FieldListLike = Union[int, FieldID, List[int], List[FieldID]]


class ExternalResources:
    def __init__(self, handle: Any) -> None:
        """
        Stores a collection of physical regions that were attached together
        using the same IndexAttach operation. Wraps a
        `legion_external_resources_t` object from the Legion C API.
        """
        self.handle = handle

    def __del__(self) -> None:
        self.destroy(unordered=True)

    def destroy(self, unordered: bool) -> None:
        """
        Eagerly destroy this object before the garbage collector does.
        It is illegal to use the object after this call.

        Parameters
        ----------
        unordered : bool
            Whether this object is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_external_resources_destroy(self.handle)
        self.handle = None


class BufferBuilder:
    def __init__(self, type_safe: bool = False) -> None:
        """
        A BufferBuilder object is a helpful utility for constructing
        buffers of bytes to pass through to tasks in other languages.
        """

        self.fmt: list[str] = []  # struct format string
        self.fmt.append("=")
        self.size = 0
        self.args: list[Union[int, float, bytes]] = []
        self.string: Optional[bytes] = None
        self.arglen: Optional[int] = None
        self.type_safe = type_safe

    def add_arg(self, arg: Union[int, float, bytes], type_val: int) -> None:
        # Save the type of the object as integer right before it
        # The integer must be matched in the C++ code in the unpack functions
        if self.type_safe:
            self.fmt.append("i")
            self.size += 4
            self.args.append(type_val)
        self.args.append(arg)

    def pack_8bit_int(self, arg: int) -> None:
        self.fmt.append("b")
        self.size += 1
        self.add_arg(arg, legion.LEGION_TYPE_INT8)

    def pack_16bit_int(self, arg: int) -> None:
        self.fmt.append("h")
        self.size += 2
        self.add_arg(arg, legion.LEGION_TYPE_INT16)

    def pack_32bit_int(self, arg: int) -> None:
        self.fmt.append("i")
        self.size += 4
        self.add_arg(arg, legion.LEGION_TYPE_INT32)

    def pack_64bit_int(self, arg: int) -> None:
        self.fmt.append("q")
        self.size += 8
        self.add_arg(arg, legion.LEGION_TYPE_INT64)

    def pack_8bit_uint(self, arg: int) -> None:
        self.fmt.append("B")
        self.size += 1
        self.add_arg(arg, legion.LEGION_TYPE_UINT8)

    def pack_16bit_uint(self, arg: int) -> None:
        self.fmt.append("H")
        self.size += 2
        self.add_arg(arg, legion.LEGION_TYPE_UINT16)

    def pack_32bit_uint(self, arg: int) -> None:
        self.fmt.append("I")
        self.size += 4
        self.add_arg(arg, legion.LEGION_TYPE_UINT32)

    def pack_64bit_uint(self, arg: int) -> None:
        self.fmt.append("Q")
        self.size += 8
        self.add_arg(arg, legion.LEGION_TYPE_UINT64)

    def pack_32bit_float(self, arg: float) -> None:
        self.fmt.append("f")
        self.size += 4
        self.add_arg(arg, legion.LEGION_TYPE_FLOAT32)

    def pack_64bit_float(self, arg: float) -> None:
        self.fmt.append("d")
        self.size += 8
        self.add_arg(arg, legion.LEGION_TYPE_FLOAT64)

    def pack_bool(self, arg: bool) -> None:
        self.fmt.append("?")
        self.size += 1
        self.add_arg(arg, legion.LEGION_TYPE_BOOL)

    def pack_16bit_float(self, arg: int) -> None:
        self.fmt.append("h")
        self.size += 2
        self.add_arg(arg, legion.LEGION_TYPE_FLOAT16)

    def pack_char(self, arg: str) -> None:
        self.fmt.append("c")
        self.size += 1
        self.add_arg(bytes(arg.encode("utf-8")), legion.LEGION_TYPE_TOTAL + 1)

    def pack_64bit_complex(self, arg: complex) -> None:
        if self.type_safe:
            self.fmt.append("i")
            self.size += 4
            self.args.append(legion.LEGION_TYPE_COMPLEX64)
        self.fmt.append("ff")  # encode complex as two floats
        self.args.append(arg.real)
        self.args.append(arg.imag)

    def pack_128bit_complex(self, arg: complex) -> None:
        if self.type_safe:
            self.fmt.append("i")
            self.size += 4
            self.args.append(legion.LEGION_TYPE_COMPLEX128)
        self.fmt.append("dd")  # encode complex as two floats
        self.args.append(arg.real)
        self.args.append(arg.imag)

    def pack_dimension(self, dim: int) -> None:
        self.pack_32bit_int(dim)

    def pack_point(self, point: Point) -> None:
        if not isinstance(point, tuple):
            raise ValueError("'point' must be a tuple")
        dim = len(point)
        if dim <= 0:
            raise ValueError("'dim' must be positive")
        if self.type_safe:
            self.pack_32bit_int(dim)
        for p in point:
            self.pack_64bit_int(p)

    def pack_accessor(
        self,
        field_id: int,
        transform: Optional[AffineTransform] = None,
        point_transform: Optional[AffineTransform] = None,
    ) -> None:
        self.pack_32bit_int(field_id)
        if not transform:
            if point_transform is not None:
                raise ValueError(
                    "'point_transform' not allowed with 'transform'"
                )
            self.pack_32bit_int(0)
        else:
            self.pack_32bit_int(transform.M)
            self.pack_32bit_int(transform.N)
            self.pack_transform(transform)
            # Pack the point transform if we have one
            if point_transform is not None:
                if transform.N != point_transform.M:
                    raise ValueError("Dimension mismatch")
                self.pack_transform(point_transform)

    def pack_transform(self, transform: AffineTransform) -> None:
        for x in range(0, transform.M):
            for y in range(0, transform.N):
                self.pack_64bit_int(transform.trans[x, y])
        for x in range(0, transform.M):
            self.pack_64bit_int(transform.offset[x])

    def pack_string(self, string: str) -> None:
        self.pack_32bit_uint(len(string))
        for char in string:
            self.pack_char(char)

    def pack_buffer(self, buf: Any) -> None:
        self.pack_32bit_uint(buf.get_size())
        self.fmt.append(buf.fmt[1:])
        self.size += buf.size
        self.args.append(*(buf.args))

    # Static member of this class for encoding dtypes
    _dtype_codes = {
        bool: legion.LEGION_TYPE_BOOL,  # same a np.bool
        np.bool_: legion.LEGION_TYPE_BOOL,
        np.int8: legion.LEGION_TYPE_INT8,
        np.int16: legion.LEGION_TYPE_INT16,
        int: legion.LEGION_TYPE_INT32,  # same as np.int
        np.int32: legion.LEGION_TYPE_INT32,
        np.int64: legion.LEGION_TYPE_INT64,
        np.uint8: legion.LEGION_TYPE_UINT8,
        np.uint16: legion.LEGION_TYPE_UINT16,
        np.uint32: legion.LEGION_TYPE_UINT32,
        np.uint64: legion.LEGION_TYPE_UINT64,
        np.float16: legion.LEGION_TYPE_FLOAT16,
        float: legion.LEGION_TYPE_FLOAT32,  # same as np.float
        np.float32: legion.LEGION_TYPE_FLOAT32,
        np.float64: legion.LEGION_TYPE_FLOAT64,
        np.complex64: legion.LEGION_TYPE_COMPLEX64,
        np.complex128: legion.LEGION_TYPE_COMPLEX128,
    }

    @classmethod
    def encode_dtype(cls, dtype: Any) -> int:
        if dtype in cls._dtype_codes:
            return cls._dtype_codes[dtype]
        elif hasattr(dtype, "type") and dtype.type in cls._dtype_codes:
            return cls._dtype_codes[dtype.type]
        raise ValueError(
            str(dtype) + " is not a valid data type for BufferBuilder"
        )

    def pack_dtype(self, dtype: Any) -> None:
        self.pack_32bit_int(self.encode_dtype(dtype))

    def get_string(self) -> Optional[bytes]:
        if self.string is None or self.arglen != len(self.args):
            fmtstr = "".join(self.fmt)
            assert len(fmtstr) == len(self.args) + 1
            self.string = struct.pack(fmtstr, *self.args)
            self.arglen = len(self.args)
        return self.string

    def get_size(self) -> int:
        return self.size


class Logger:
    def __init__(self, name: str) -> None:
        self.handle = legion.legion_logger_create(name.encode("utf-8"))

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        """
        Eagerly destroy this object before the garbage collector does.
        It is illegal to use the object after this call.
        """
        if self.handle is None:
            return
        legion.legion_logger_destroy(self.handle)
        self.handle = None

    def spew(self, msg: str) -> None:
        if self.want_spew:
            legion.legion_logger_spew(self.handle, msg.encode("utf-8"))

    def debug(self, msg: str) -> None:
        if self.want_debug:
            legion.legion_logger_debug(self.handle, msg.encode("utf-8"))

    def info(self, msg: str) -> None:
        if self.want_info:
            legion.legion_logger_info(self.handle, msg.encode("utf-8"))

    def print(self, msg: str) -> None:
        if self.want_print:
            legion.legion_logger_print(self.handle, msg.encode("utf-8"))

    def warning(self, msg: str) -> None:
        if self.want_warning:
            legion.legion_logger_warning(self.handle, msg.encode("utf-8"))

    def error(self, msg: str) -> None:
        if self.want_error:
            legion.legion_logger_error(self.handle, msg.encode("utf-8"))

    def fatal(self, msg: str) -> None:
        if self.want_fatal:
            legion.legion_logger_fatal(self.handle, msg.encode("utf-8"))

    @cached_property
    def want_spew(self) -> bool:
        return legion.legion_logger_want_spew(self.handle)

    @cached_property
    def want_debug(self) -> bool:
        return legion.legion_logger_want_debug(self.handle)

    @cached_property
    def want_info(self) -> bool:
        return legion.legion_logger_want_info(self.handle)

    @cached_property
    def want_print(self) -> bool:
        return legion.legion_logger_want_print(self.handle)

    @cached_property
    def want_warning(self) -> bool:
        return legion.legion_logger_want_warning(self.handle)

    @cached_property
    def want_error(self) -> bool:
        return legion.legion_logger_want_error(self.handle)

    @cached_property
    def want_fatal(self) -> bool:
        return legion.legion_logger_want_fatal(self.handle)
