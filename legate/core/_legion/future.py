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

from typing import Any, Optional

from .. import ffi, legion
from .geometry import Point, Rect
from .pending import _pending_deletions


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
    def same_handle(self, other: Any) -> bool:
        return (  # noqa
            type(self) == type(other) and self.handle == other.handle
        )

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
        self,
        runtime: legion.legion_runtime_t,
        data: Any,
        size: int,
        shard_local: bool = False,
        type: Optional[Any] = None,
        provenance: Optional[str] = None,
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
        if provenance is None:
            provenance = ""
        if self.handle is not None:
            raise RuntimeError("Future must be unset to set its value")
        self.handle = legion.legion_future_from_untyped_pointer_detailed(
            runtime,
            ffi.from_buffer(data),
            size,
            False,
            provenance.encode("ascii"),
            shard_local,
        )
        self._type = type

    @classmethod
    def from_buffer(
        cls,
        runtime: legion.legion_runtime_t,
        buf: Any,
        shard_local: bool = False,
        type: Optional[Any] = None,
        provenance: Optional[str] = None,
    ) -> Future:
        """
        Construct a future from a buffer storing data

        Parameters
        ----------
        buf : buffer
            Buffer to create a future from
        Returns
        -------
        Future
        """
        if provenance is None:
            provenance = ""
        return cls(
            legion.legion_future_from_untyped_pointer_detailed(
                runtime,
                ffi.from_buffer(buf),
                len(buf),
                False,
                provenance.encode("ascii"),
                shard_local,
            ),
            type=type,
        )

    @classmethod
    def from_cdata(
        cls,
        runtime: legion.legion_runtime_t,
        cdata: Any,
        shard_local: bool = False,
        type: Optional[Any] = None,
    ) -> Future:
        return cls.from_buffer(
            runtime,
            ffi.buffer(ffi.addressof(cdata)),
            shard_local=shard_local,
            type=type,
        )

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
        if self.handle is None:
            return True
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
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        redop: int,
        ordered: bool = True,
        mapper: int = 0,
        tag: int = 0,
        init_value: Optional[Future] = None,
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
        ordered : bool
            If ``True``, reductions are performed in an ordered manner
            so the result is deterministic
        mapper : int
            ID of the mapper for managing the mapping of the task
        tag : int
            Tag to pass to the mapper to provide calling context
        init_value : Future
            Optional future holding the initial value for reductions

        Returns
        -------
        Future representing the reduced value of all the future in the map
        """
        # TODO: We need to pass a meaningful provenance string intead of
        # defaulting to an empty string
        if init_value is not None:
            return Future(
                legion.legion_future_map_reduce_with_initial_value(
                    runtime,
                    context,
                    self.handle,
                    redop,
                    ordered,
                    mapper,
                    tag,
                    "".encode(),
                    init_value.handle,
                )
            )

        return Future(
            legion.legion_future_map_reduce(
                runtime,
                context,
                self.handle,
                redop,
                ordered,
                mapper,
                tag,
            )
        )

    @classmethod
    def from_list(
        cls,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        futures: list[Future],
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
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
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
        for i, (point, future) in enumerate(futures.items()):
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
