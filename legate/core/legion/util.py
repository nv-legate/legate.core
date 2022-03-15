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
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np

from legion_cffi import ffi, lib as legion

if TYPE_CHECKING:
    from ..context import Context
    from ..runtime import Runtime
    from . import (
        AffineTransform,
        ArgumentMap,
        Detach,
        FieldSpace,
        IndexPartition,
        IndexSpace,
        OutputRegion,
        PhysicalRegion,
        Point,
        Rect,
        Region,
    )

# We can't call out to the CFFI from inside of finalizer methods
# because that can risk a deadlock (CFFI's lock is stupid, they
# take it still in python so if a garbage collection is triggered
# while holding it you can end up deadlocking trying to do another
# CFFI call inside a finalizer because the lock is not reentrant).
# Therefore we defer deletions until we end up launching things
# later at which point we know that it is safe to issue deletions
_pending_unordered: dict[Any, Any] = dict()

# We also have some deletion operations which are only safe to
# be done if we know the Legion runtime is still running so we'll
# move them here and only issue the when we know we are inside
# of the execution of a task in some way
_pending_deletions: list[Any] = list()


def legate_task_preamble(runtime: Runtime, context: Context) -> None:
    """
    This function sets up internal Legate state for a task in Python.
    In general, users only need to worry about calling this function
    at the beginning of sub-tasks on the Python side. The Legate
    Core will perform the necessary call to this function for the
    top-level task.
    """
    assert context not in _pending_unordered
    _pending_unordered[context] = list()


def legate_task_progress(runtime: Runtime, context: Context) -> None:
    """
    This method will progress any internal Legate Core functionality
    that is running in the background. Legate clients do not need to
    call it, but can optionally do so to speed up collection.
    """
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


def legate_task_postamble(runtime: Runtime, context: Context) -> None:
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
        launcher: Any, runtime: Runtime, context: Context, *args: Any
    ) -> Any:
        # This context should always be in the dictionary
        legate_task_progress(runtime, context)
        return func(launcher, runtime, context, *args)

    return launch


class FieldID:
    def __init__(self, field_space: FieldSpace, fid: int, type: Any) -> None:
        """
        A FieldID class wraps a `legion_field_id_t` in the Legion C API.
        It provides a canonical way to represent an allocated field in a
        field space and means by which to deallocate the field.

        Parameters
        ----------
        field_space : FieldSpace
            The owner field space for this field
        fid : int
            The ID for this field
        type : type
            The type of this field
        """
        self.field_space = field_space
        self._type = type
        self.field_id = fid

    def destroy(self, unordered: bool = False) -> None:
        """
        Deallocate this field from the field space
        """
        self.field_space.destroy_field(self.field_id, unordered)

    @property
    def fid(self) -> int:
        return self.field_id

    @property
    def type(self) -> Any:
        return self._type


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


class Future:
    def __init__(
        self, handle: Optional[Any] = None, type: Optional[Any] = None
    ) -> None:
        """
        A Future object represents a pending computation from a task or other
        operation. Futures can carry "unstructured" data as a buffer of bytes
        or they can be empty and used only for synchronization.

        Parameters
        ----------
        handle : legion_future_t
            Wrap an optional handle in this Future. The Future object
            will take ownership of this handle
        type : object
            Optional object to represent the type of this future
        """
        self.handle = handle
        self._type = type

    def __del__(self) -> None:
        self.destroy(unordered=True)

    # We cannot use this as __eq__ because then we would have to define a
    # compatible __hash__, which would not be sound because self.handle can
    # change during the lifetime of a Future object, and thus so would the
    # object's hash. So we just leave the default `f1 == f2 <==> f1 is f2`.
    def same_handle(self, other: Future) -> bool:
        return type(self) == type(other) and self.handle == other.handle

    def __str__(self) -> str:
        if self.handle:
            return f"Future({str(self.handle.impl)[16:-1]})"
        return "Future(None)"

    def destroy(self, unordered: bool) -> None:
        """
        Eagerly destroy this Future before the garbage collector does
        It is illegal to use the Future after this call

        Parameters
        ----------
        unordered : bool
            Whether this Future is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_future_destroy(self.handle)
        self.handle = None

    def set_value(
        self, runtime: Runtime, data: Any, size: int, type: object = None
    ) -> None:
        """
        Parameters
        ----------
        runtime : legion_runtime_t*
            Pointer to the Legion runtime object
        data : buffer
            Set the value of the future from a buffer
        size : int
            Size of the buffer in bytes
        type : object
            An optional object to represent the type of the future
        """
        if self.handle is not None:
            raise RuntimeError("Future must be unset to set its value")
        self.handle = legion.legion_future_from_untyped_pointer(
            runtime, ffi.from_buffer(data), size
        )
        self._type = type

    def get_buffer(self, size: Optional[int] = None) -> Any:
        """
        Return a buffer storing the data for this Future.
        This will block until the future completes if it has not already.

        Parameters
        ----------
        size : int
            Optional expected size of the future
        Returns
        -------
        An object that implements the Python buffer protocol
        that contains the data
        """
        if size is None:
            size = self.get_size()
        return ffi.buffer(
            legion.legion_future_get_untyped_pointer(self.handle), size
        )

    def get_size(self) -> int:
        """
        Return the size of the buffer that the future stores.
        This will block until the future completes if it has not already.
        """
        return legion.legion_future_get_untyped_size(self.handle)

    def get_string(self) -> bytes:
        """
        Return the result of the future interpreted as a string.
        This will block until the future completes if it has not already.
        """
        size = self.get_size()
        return ffi.unpack(
            ffi.cast(
                "char *", legion.legion_future_get_untyped_pointer(self.handle)
            ),
            size,
        )

    def is_ready(self, subscribe: bool = False) -> bool:
        """
        Parameters
        ----------
        subscribe : bool
            Whether the data for this future is ultimately needed locally

        Returns
        -------
        bool indicating if the future has completed or not
        """
        return legion.legion_future_is_ready_subscribe(self.handle, subscribe)

    def wait(self) -> None:
        """
        Block waiting for the future to complete
        """
        legion.legion_future_get_void_result(self.handle)

    @property
    def type(self) -> Any:
        return self._type


class FutureMap:
    def __init__(self, handle: Optional[Any] = None) -> None:
        """
        A FutureMap object represents a collection of Future objects created by
        an index space operation such as an IndexTask launch. Applications can
        use a future map to synchronize with all the individual operations (not
        recommended), or to obtain individual futures from point operations in
        the index space launch.

        Parameters
        ----------
        handle : legion_future_map_t
            The handle for this FutureMap to wrap and take ownership of
        """
        self.handle = handle

    def __del__(self) -> None:
        self.destroy(unordered=True)

    def destroy(self, unordered: bool) -> None:
        """
        Eagerly destroy this FutureMap before the garbage collector does
        It is illegal to use the FutureMap after this call

        Parameters
        ----------
        unordered : bool
            Whether this FutureMap is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_future_map_destroy(self.handle)
        self.handle = None

    def wait(self) -> None:
        """
        Wait for all the futures in the future map to complete
        """
        legion.legion_future_map_wait_all_results(self.handle)

    def get_future(self, point: Point) -> Future:
        """
        Extract a specific future from the future map

        Parameters
        ----------
        point : Point
            The particular point in the index space launch to extract

        Returns
        -------
        Future describing the result from the particular point operation
        """
        return Future(
            legion.legion_future_map_get_future(self.handle, point.raw())
        )

    def reduce(
        self,
        context: Context,
        runtime: Runtime,
        redop: int,
        deterministic: bool = False,
        mapper: int = 0,
        tag: int = 0,
    ) -> Future:
        """
        Reduce all the futures in the future map down to a single
        future value using a reduction operator.

        Parameters
        ----------
        context : legion_context_t
            The context handle for the enclosing parent task
        runtime : legion_runtime_t
            The Legion runtime handle
        redop : int
            ID for the reduction operator to use for reducing futures
        deterministic : bool
            Whether this reduction needs to be performed deterministically
        mapper : int
            ID of the mapper for managing the mapping of the task
        tag : int
            Tag to pass to the mapper to provide calling context

        Returns
        -------
        Future representing the reduced value of all the future in the map
        """
        return Future(
            legion.legion_future_map_reduce(
                runtime,
                context,
                self.handle,
                redop,
                deterministic,
                mapper,
                tag,
            )
        )

    @classmethod
    def from_list(
        cls, context: Context, runtime: Runtime, futures: list[Future]
    ) -> Any:
        """
        Construct a FutureMap from a list of futures

        Parameters
        ----------
        context : legion_context_t
            The context handle for the enclosing parent task
        runtime : legion_runtime_t
            The Legion runtime handle
        futures : List[Future]
            A list of futures to use to construct a future map

        Returns
        -------
        FutureMap that contains all the futures in 1-D space of points
        """
        num_futures = len(futures)
        domain = Rect([num_futures]).raw()
        points = ffi.new("legion_domain_point_t[%d]" % num_futures)
        futures_ = ffi.new("legion_future_t[%d]" % num_futures)
        for i in range(num_futures):
            points[i] = Point([i]).raw()
            futures_[i] = futures[i].handle
        handle = legion.legion_future_map_construct_from_futures(
            runtime,
            context,
            domain,
            points,
            futures,
            num_futures,
            False,
            0,
            False,
        )
        return cls(handle)

    @classmethod
    def from_dict(
        cls,
        context: Context,
        runtime: Runtime,
        domain: Rect,
        futures: dict[Point, Future],
        collective: bool = False,
    ) -> Any:
        """
        Construct a FutureMap from a Point-to-Future dict

        Parameters
        ----------
        context : legion_context_t
            The context handle for the enclosing parent task
        runtime : legion_runtime_t
            The Legion runtime handle
        domain : Rect
            A dense Rect enumerating all the Futures that will be included in
            the created future map
        futures : dict[Point, Future]
            Futures to use to construct a future map
        collective : bool
            If True then each shard can specify a different subset of the
            Futures to include. The runtime will combine all the Futures
            provided by the different shards into a single future map.

        Returns
        -------
        FutureMap that contains all the Futures
        """
        num_futures = len(futures)
        points = ffi.new("legion_domain_point_t[%d]" % num_futures)
        futures_ = ffi.new("legion_future_t[%d]" % num_futures)
        for (i, (point, future)) in enumerate(futures.items()):
            points[i] = point.raw()
            futures_[i] = future.handle
        handle = legion.legion_future_map_construct_from_futures(
            runtime,
            context,
            domain.raw(),
            points,
            futures_,
            num_futures,
            collective,
            0,
            True,
        )
        return cls(handle)


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

    def pack_value(self, value: Any, val_type: Any) -> None:
        if np.dtype(val_type) == np.int16:
            self.pack_16bit_int(value)
        elif np.dtype(val_type) == np.int32:
            self.pack_32bit_int(value)
        elif np.dtype(val_type) == np.int64:
            self.pack_64bit_int(value)
        elif np.dtype(val_type) == np.uint16:
            self.pack_16bit_uint(value)
        elif np.dtype(val_type) == np.uint32:
            self.pack_32bit_uint(value)
        elif np.dtype(val_type) == np.uint64:
            self.pack_64bit_uint(value)
        elif np.dtype(val_type) == np.float32:
            self.pack_32bit_float(value)
        elif np.dtype(val_type) == np.float64:
            self.pack_64bit_float(value)
        elif np.dtype(val_type) == bool:  # np.bool
            self.pack_bool(value)
        elif np.dtype(val_type) == np.float16:
            self.pack_16bit_float(value)
        elif np.dtype(val_type) == np.complex64:
            self.pack_64bit_complex(value)
        elif np.dtype(val_type) == np.complex128:
            self.pack_128bit_complex(value)
        else:
            raise TypeError("Unhandled value type")

    def pack_string(self, string: str) -> None:
        self.pack_32bit_int(len(string))
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
