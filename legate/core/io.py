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

from typing import Iterable, Tuple, Union

from legion_cffi import ffi  # Make sure we only have one ffi instance

from .legate import Array, Table
from .legion import (
    FutureMap,
    IndexPartition,
    PartitionByDomain,
    Point,
    Rect,
    legion,
)
from .partition import Tiling
from .runtime import _runtime
from .shape import Shape
from .store import DistributedAllocation, Store


class DataSplit(object):
    """
    Objects of this class can be used to describe the existing partitioning
    of incoming data to the `ingest` call. This is an abstract base class,
    use one the concrete derived classes instead.
    """

    def make_partition(
        self,
        store: Store,
        colors: Union[int, Tuple[int]],
        local_colors: Iterable[Point],
    ):
        raise NotImplementedError("Implement in derived classes")


class CustomSplit(DataSplit):
    def __init__(self, get_subdomain):
        """
        Used to describe an arbitrary partitioning of the incoming data. Note
        that Legate may not be able to effectively re-use this arbitrary
        partition, and thus will likely need to reshuffle the data before use.

        Parameters
        ----------
        get_subdomain : Callable[[Point], Rect]
            Function that returns the subset of the overall domain covered by
            the buffer of the corresponding color. This function will only be
            called once for each color, on the appropriate process for that
            color (see the documentation for `get_local_colors` on `ingest`).
        """
        self.get_subdomain = get_subdomain

    def make_partition(self, store, colors, local_colors):
        fut_size = ffi.sizeof("legion_domain_t")
        futures = {}
        for c in local_colors:
            rect = self.get_subdomain(c)
            futures[c] = _runtime.create_future(
                ffi.buffer(ffi.addressof(rect.raw())), fut_size
            )
        domains = FutureMap.from_dict(
            _runtime.legion_context,
            _runtime.legion_runtime,
            Rect(colors),
            futures,
            collective=True,
        )
        region = store.storage.region
        index_partition = IndexPartition(
            _runtime.legion_context,
            _runtime.legion_runtime,
            region.index_space,
            _runtime.find_or_create_index_space(colors),
            PartitionByDomain(domains),
        )
        return region.get_child(index_partition)


class TiledSplit(DataSplit):
    def __init__(self, tile_shape):
        """
        Used to describe a tiling of the domain, where tiles are all of equal
        size, and packed according to color order.

        Parameters
        ----------
        tile_shape : int | Tuple[int]
            The shape of each tile
        """
        self.tile_shape = tile_shape

    def make_partition(self, store, colors, local_colors):
        functor = Tiling(
            _runtime,
            Shape(self.tile_shape),
            Shape(colors),
        )
        store.set_key_partition(functor)
        part = store.find_or_create_legion_partition(functor, complete=True)
        assert store.compute_projection() == 0
        return part


def ingest(
    dtype, shape, colors, data_split, get_buffer, get_local_colors=None
):
    """
    Construct a single-column Table backed by a collection of buffers
    distributed across the machine.

    Each buffer is assumed to cover a disjoint dense subset of a rectangular
    n-dimensional domain, and is identified by its "color", an m-dimensional
    integer point.

    Parameters
    ----------
    dtype : pyarrow.DataType
        Type of the data to ingest

    shape : int | Tuple[int]
        N-dimensional dense rectangular domain of the data to ingest

    colors : int | Tuple[int]
        M-dimensional dense rectangle indexing all the buffers to ingest

    data_split : DataSplit
        Specifies what subset of the overall domain is covered by each buffer

    get_buffer : Callable[[Point], memoryview]
        This function will be called on the appropriate process for each color
        (see the documentation for `get_local_colors`) and should return a
        pre-existing buffer residing in local memory, or generate one on the
        fly, e.g. by reading from a file.

        The contents of each buffer may be in row-major or column-major order.
        Legate will take ownership of the returned memory.

    get_local_colors : Callable[[], Iterable[Point]] | None
        If `None` then Legate will assume that every buffer is accessible from
        any process (rank) where Legate is running. Legate will then invoke
        `get_buffer` once for each color, with each invocation happening on an
        unspecified process and host, wherever Legate decides it wants that
        subset of the domain to reside.

        This mode is appropriate e.g. if each buffer is created on the fly by
        loading from a file, and every file is available on all hosts over a
        distributed filesystem.

        If not `None`, then Legate will assume that each buffer is only
        available on a single process. It will invoke `get_local_colors` once
        on each process, and will only inquire about the returned colors on
        that process. Each color in `colors` should be returned from exactly
        one invocation of `get_local_colors`.

        If you are launching Legate by specifiying a `--launcher` flag to the
        Legate driver, then by default every Legate process corresponds to a
        different host, and you can use the hostname to decide which buffers
        every call to `get_buffer` should return. If you are performing a
        custom launch then it is possible that multiple Legate processes are
        running on the same host, in which case the hostname will not be
        sufficient, and you will need to consult other information, e.g. the
        process ID or a launcher-set environment variable such as
        `OMPI_COMM_WORLD_RANK`.

        This mode is appropriate e.g. if you want to ingest the pre-distributed
        output of a preceding non-Legate computation, or if the data is loaded
        from a collection of files, each of which is only available on a single
        host. Because you are essentially dictating the placement of data under
        this mode, it is more likely that Legate will need to reshuffle the
        data to meet the needs of a subsequent operation.

    Returns
    -------
    A single-column Table backed by the provided buffers
    """
    if not isinstance(data_split, DataSplit):
        raise TypeError(
            f"data_split: expected a DataSplit object but got {data_split}"
        )
    if get_local_colors is None:

        # Assign colors following the default sharding
        def get_local_colors():
            sid = _runtime.core_context.get_sharding_id(
                _runtime.core_library.LEGATE_CORE_LINEARIZE_SHARD_ID
            )
            shard = legion.legion_runtime_local_shard(
                _runtime.legion_runtime, _runtime.legion_context
            )
            domain = Rect(colors).raw()
            total_shards = legion.legion_runtime_total_shards(
                _runtime.legion_runtime, _runtime.legion_context
            )
            points_size = ffi.new("size_t *")
            points_size[0] = 1
            for c in colors:
                points_size[0] *= c
            points_ptr = ffi.new("legion_domain_point_t[%s]" % points_size[0])
            legion.legion_sharding_functor_invert(
                sid,
                shard,
                domain,
                domain,
                total_shards,
                points_ptr,
                points_size,
            )
            points = []
            for i in range(points_size[0]):
                points.append(Point(points_ptr[i]))
            return points

    store = _runtime.core_context.create_store(dtype, shape)
    local_colors = get_local_colors()
    partition = data_split.make_partition(store, colors, local_colors)
    shard_local_buffers = {c: get_buffer(c) for c in local_colors}
    alloc = DistributedAllocation(partition, shard_local_buffers)
    store.attach_external_allocation(_runtime.core_context, alloc, False)
    # first store is the (non-existent) mask
    array = Array(dtype, [None, store])
    return Table.from_arrays([array], ["ingested"])
