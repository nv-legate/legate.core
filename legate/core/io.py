# Copyright 2021 NVIDIA Corporation
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

from legion_cffi import ffi  # Make sure we only have one ffi instance

from .legion import Point, Rect, legion
from .runtime import _runtime
from .store import DistributedAllocation


def ingest(dtype, shape, colors, get_buffer, get_local_colors=None):
    """
    Construct a Store backed by a collection of buffers distributed across the
    machine.

    Each buffer is assumed to cover a disjoint dense subset of a rectangular
    n-dimensional domain, and is identified by its "color", an m-dimensional
    integer point. The contents of each buffer are assumed to be in row-major
    order.

    Parameters
    ----------
    dtype : numpy.dtype
        Type of the data to ingest

    shape : int | Tuple[int]
        N-dimensional dense rectangular domain of the data to ingest

    colors : int | Tuple[int]
        M-dimensional dense rectangle indexing all the buffers to ingest

    get_buffer: Callable[[Point], Tuple[Rect,memoryview]]
        This function will be called on the appropriate process for each color
        (see the documentation for `get_local_colors`), and should produce a
        `(subdomain,buffer)` tuple, where `subdomain` is the subset of the
        overall domain covered by `buffer`. It may return a pre-existing buffer
        residing in local memory, or generate one on the fly, e.g. by reading
        from a file. Legate will take ownership of this memory.

    get_local_colors: Callable[[], Iterable[Point]] | None
        If `None` then Legate will assume that every buffer is accessible from
        any process where Legate is running. Legate will then invoke
        `get_buffer` once for each color, with each invocation happening on an
        unspecified process and host, wherever Legate decides it wants that
        subset of the domain to reside.

        This mode is appropriate e.g. if each buffer is created on the fly by
        loading from a file, and every file is available on all hosts over a
        distributed filesystem.

        If not `None`, then Legate will assume that each buffer is only
        available on a single process. It will invoke `get_local_colors` once
        on each process, and subsequently invoke `get_buffer` on that process
        for each color returned. Each color in `colors` should be returned from
        exactly one invocation of `get_local_colors`.

        If you are launching Legate by specifiying a `--launcher` flag to the
        Legate driver, then every Legate process corresponds to a different
        host, and you can use the hostname to decide which buffers every call
        to `get_buffer` should return. If you are performing a custom launch
        then it is possible that multiple Legate processes are running on the
        same host, in which case the hostname will not be sufficient, and you
        will need to consult other information, e.g. the process ID or a
        launcher-set environment variable such as `OMPI_COMM_WORLD_RANK`.

        This mode is appropriate e.g. if you want to ingest the pre-distributed
        output of a preceding non-Legate computation, or if the data is loaded
        from a collection of files, each of which is only available on a single
        host. Because you are essentially dictating the placement of data under
        this mode, it is more likely that Legate will need to reshuffle the
        data to meet the needs of a subsequent operation.

    Returns
    -------
    A Store backed by the provided buffers
    """
    colors = Rect(colors)
    if get_local_colors is None:

        def get_local_colors():
            sid = _runtime.core_context.get_sharding_id(
                _runtime.core_library.LEGATE_CORE_LINEARIZE_SHARD_ID
            )
            shard = legion.legion_runtime_local_shard(
                _runtime.legion_runtime, _runtime.legion_context
            )
            domain = colors.raw()
            total_shards = legion.legion_runtime_total_shards(
                _runtime.legion_runtime, _runtime.legion_context
            )
            points_size = ffi.new("size_t *")
            points_size[0] = colors.get_volume()
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

    shard_local_domains = {}
    shard_local_buffers = {}
    for c in get_local_colors():
        rect, buf = get_buffer(c)
        shard_local_domains[c] = rect
        shard_local_buffers[c] = buf
    alloc = DistributedAllocation(
        colors, shard_local_domains, shard_local_buffers
    )
    store = _runtime.core_context.create_store(dtype, shape)
    store.attach_external_allocation(_runtime.core_context, alloc, False)
    return store
