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

import gc
import math
import struct
from collections import deque
from functools import reduce

from legion_cffi import ffi  # Make sure we only have one ffi instance
from legion_top import cleanup_items, top_level

from legate.core import types as ty

from .context import Context
from .corelib import CoreLib
from .legion import (
    FieldSpace,
    Future,
    IndexSpace,
    OutputRegion,
    Rect,
    Region,
    legate_task_postamble,
    legate_task_preamble,
    legion,
)
from .partition import Restriction
from .shape import Shape
from .solver import Partitioner
from .store import RegionField, Store


# A Field holds a reference to a field in a region tree
# that can be used by many different RegionField objects
class Field(object):
    __slots__ = [
        "runtime",
        "region",
        "field_id",
        "dtype",
        "shape",
        "own",
        "field_space",
    ]

    def __init__(
        self,
        runtime,
        region,
        field_id,
        dtype,
        shape,
        own=True,
    ):
        self.runtime = runtime
        self.region = region
        self.field_id = field_id
        self.dtype = dtype
        self.shape = shape
        self.own = own

    def __del__(self):
        if self.own:
            # Return our field back to the runtime
            self.runtime.free_field(
                self.region,
                self.field_id,
                self.dtype,
                self.shape,
            )


_sizeof_int = ffi.sizeof("int")
_sizeof_size_t = ffi.sizeof("size_t")
assert _sizeof_size_t == 4 or _sizeof_size_t == 8


# A helper class for doing field management with control replication
class FieldMatch(object):
    __slots__ = ["manager", "fields", "input", "output", "future"]

    def __init__(self, manager, fields):
        self.manager = manager
        self.fields = fields
        # Allocate arrays of ints that are twice as long as fields since
        # our values will be 'field_id,tree_id' pairs
        if len(fields) > 0:
            alloc_string = "int[" + str(2 * len(fields)) + "]"
            self.input = ffi.new(alloc_string)
            self.output = ffi.new(alloc_string)
            # Fill in the input buffer with our data
            for idx in range(len(fields)):
                region, field_id = fields[idx]
                self.input[2 * idx] = region.handle.tree_id
                self.input[2 * idx + 1] = field_id
        else:
            self.input = ffi.NULL
            self.output = ffi.NULL
        self.future = None

    def launch(self, runtime, context):
        assert self.future is None
        self.future = Future(
            legion.legion_context_consensus_match(
                runtime,
                context,
                self.input,
                self.output,
                len(self.fields),
                2 * _sizeof_int,
            )
        )
        return self.future

    def update_free_fields(self):
        # If we know there are no fields then we can be done early
        if len(self.fields) == 0:
            return
        # Wait for the future to be ready
        if not self.future.is_ready():
            self.future.wait()
        # Get the size of the buffer in the returned
        if _sizeof_size_t == 4:
            num_fields = struct.unpack_from("I", self.future.get_buffer(4))[0]
        else:
            num_fields = struct.unpack_from("Q", self.future.get_buffer(8))[0]
        assert num_fields <= len(self.fields)
        if num_fields > 0:
            # Put all the returned fields onto the ordered queue in the order
            # that they are in this list since we know
            ordered_fields = [None] * num_fields
            for region, field_id in self.fields:
                found = False
                for idx in range(num_fields):
                    if self.output[2 * idx] != region.handle.tree_id:
                        continue
                    if self.output[2 * idx + 1] != field_id:
                        continue
                    assert ordered_fields[idx] is None
                    ordered_fields[idx] = (region, field_id)
                    found = True
                    break
                if not found:
                    # Not found so put it back int the unordered queue
                    self.manager.free_field(region, field_id, ordered=False)
            # Notice that we do this in the order of the list which is the
            # same order as they were in the array returned by the match
            for region, field_id in ordered_fields:
                self.manager.free_field(region, field_id, ordered=True)
        else:
            # No fields on all shards so put all our fields back into
            # the unorered queue
            for region, field_id in self.fields:
                self.manager.free_field(region, field_id, ordered=False)


# This class manages all regions of the same shape.
class RegionManager(object):
    def __init__(self, runtime, shape):
        self._runtime = runtime
        self._shape = shape
        self._top_regions = []
        self._region_set = set()

    def __del__(self):
        self._shape = None
        self._top_regions = None
        self._region_set = None

    def destroy(self):
        while self._top_regions:
            region = self._top_regions.pop()
            region.destroy()
        self._top_regions = []
        self._region_set = set()

    def import_region(self, region):
        if region not in self._region_set:
            self._top_regions.append(region)
            self._region_set.add(region)

    @property
    def active_region(self):
        return self._top_regions[-1]

    @property
    def has_space(self):
        return (
            len(self._top_regions) > 0
            and self.active_region.field_space.has_space
        )

    def _create_region(self):
        # Note that the regions created in this method are always fresh
        # so we don't need to de-duplicate them to keep track of their
        # life cycles correctly.
        index_space = self._shape.get_index_space(self._runtime)
        field_space = self._runtime.create_field_space()
        region = self._runtime.create_region(index_space, field_space)
        self._top_regions.append(region)
        self._region_set.add(region)

    def allocate_field(self, dtype):
        if not self.has_space:
            self._create_region()
        region = self.active_region
        field_id = region.field_space.allocate_field(dtype)
        return region, field_id


# This class manages the allocation and reuse of fields
class FieldManager(object):
    def __init__(self, runtime, shape, dtype):
        self.runtime = runtime
        self.shape = shape
        self.dtype = dtype
        # This is a sanitized list of (region,field_id) pairs that is
        # guaranteed to be ordered across all the shards even with
        # control replication
        self.free_fields = deque()
        # This is an unsanitized list of (region,field_id) pairs which is not
        # guaranteed to be ordered across all shards with control replication
        self.freed_fields = list()
        # A list of match operations that have been issued and for which
        # we are waiting for values to come back
        self.matches = deque()
        self.match_counter = 0
        # Figure out how big our match frequency is based on our size
        volume = reduce(lambda x, y: x * y, self.shape)
        size = volume * self.dtype.size
        if size > runtime.max_field_reuse_size:
            # Figure out the ratio our size to the max reuse size (round up)
            ratio = (
                size + runtime.max_field_reuse_size - 1
            ) // runtime.max_field_reuse_size
            assert ratio >= 1
            # Scale the frequency by the ratio, but make it at least 1
            self.match_frequency = (
                runtime.max_field_reuse_frequency + ratio - 1
            ) // ratio
        else:
            self.match_frequency = runtime.max_field_reuse_frequency

    def destroy(self):
        self.free_fields = None
        self.freed_fields = None
        self.fill_space = None

    def allocate_field(self):
        # Increment our match counter
        self.match_counter += 1
        # If the match counter equals our match frequency then do an exchange
        if self.match_counter == self.match_frequency:
            # This is where the rubber meets the road between control
            # replication and garbage collection. We need to see if there
            # are any freed fields that are shared across all the shards.
            # We have to test this deterministically no matter what even
            # if we don't have any fields to offer ourselves because this
            # is a collective with other shards. If we have any we can use
            # the first one and put the remainder on our free fields list
            # so that we can reuse them later. If there aren't any then
            # all the shards will go allocate a new field.
            local_freed_fields = self.freed_fields
            # The match now owns our freed fields so make a new list
            # Have to do this before dispatching the match
            self.freed_fields = list()
            match = FieldMatch(self, local_freed_fields)
            # Dispatch the match
            self.runtime.dispatch(match)
            # Put it on the deque of outstanding matches
            self.matches.append(match)
            # Reset the match counter back to 0
            self.match_counter = 0
        # First, if we have a free field then we know everyone has one of those
        if len(self.free_fields) > 0:
            return self.free_fields.popleft()
        # If we don't have any free fields then see if we have a pending match
        # outstanding that we can now add to our free fields and use
        while len(self.matches) > 0:
            match = self.matches.popleft()
            match.update_free_fields()
            # Check again to see if we have any free fields
            if len(self.free_fields) > 0:
                return self.free_fields.popleft()

        region_manager = self.runtime.find_or_create_region_manager(self.shape)
        return region_manager.allocate_field(self.dtype)

    def free_field(self, region, field_id, ordered=False):
        if ordered:
            if self.free_fields is not None:
                self.free_fields.append((region, field_id))
        else:  # Put this on the unordered list
            if self.freed_fields is not None:
                self.freed_fields.append((region, field_id))


class ExternalAllocation(object):
    """
    Any external allocation that a client library wants to attach to
    a Legate store must be wrapped by an instance of the ExternalAllocation
    interface. Legate uses this custom interface instead of Python's
    memoryview interface because it needs to know the exact starting
    address of the allocation for the alias analysis; an external
    allocation attached to more than one Legate store can lead to
    all sorts of undefined behaviors.
    """

    @property
    def address(self):
        raise NotImplementedError("Should be implemented by a subclass")

    @property
    def memoryview(self):
        raise NotImplementedError("Should be implemented by a subclass")


class Attachment(object):
    def __init__(self, ptr, extent, region, field):
        self.ptr = ptr
        self.extent = extent
        self.end = ptr + extent - 1
        self.count = 1
        self.region = region
        self.field = field

    def overlaps(self, other):
        return not (self.end < other.ptr or other.end < self.ptr)

    def equals(self, other):
        # Sufficient to check the pointer and extent
        # as they are used as a key for de-duplication
        return self.ptr == other.ptr and self.extent == other.extent

    def add_reference(self):
        self.count += 1

    def remove_reference(self):
        assert self.count > 0
        self.count += 1

    @property
    def collectible(self):
        return self.count == 0


class AttachmentManager(object):
    def __init__(self, runtime):
        self._runtime = runtime

        self._attachments = dict()

        self._next_detachment_key = 0
        self._registered_detachments = dict()
        self._deferred_detachments = list()
        self._pending_detachments = dict()

    def destroy(self):
        gc.collect()
        while self._deferred_detachments:
            self.perform_detachments()
            # Make sure progress is made on any of these operations
            self._runtime._progress_unordered_operations()
            gc.collect()
        # Always make sure we wait for any pending detachments to be done
        # so that we don't lose the references and make the GC unhappy
        gc.collect()
        while self._pending_detachments:
            self.prune_detachments()
            gc.collect()

        # Clean up our attachments so that they can be collected
        self._attachments = None

    @staticmethod
    def attachment_key(alloc):
        return (alloc.address, alloc.memoryview.nbytes)

    def has_attachment(self, array):
        key = self.attachment_key(array)
        return key in self._attachments

    def attach_external_allocation(self, context, alloc, shape, dtype, share):
        key = self.attachment_key(alloc)
        if key not in self._attachments:
            region_field = self._runtime.allocate_field(shape, dtype)
            region_field.attach_external_allocation(context, alloc, share)
            attachment = Attachment(
                *key, region_field.region, region_field.field
            )

            # iterate over attachments and look for aliases which are bad
            for other in self._attachments.values():
                if other.overlaps(attachment):
                    assert not other.equals(attachment)
                    raise RuntimeError(
                        "Illegal aliased attachments not supported by Legate"
                    )

            self._attachments[key] = attachment
        else:
            attachment = self._attachments[key]
            attachment.add_reference()
            region = attachment.region
            field = attachment.field
            region_field = RegionField(self._runtime, region, field, shape)
        return self._runtime.create_store(dtype, shape, storage=region_field)

    def remove_attachment(self, alloc):
        key = self.attachment_key(alloc)
        if key not in self._attachments:
            raise RuntimeError("Unable to find attachment to remove")
        attachment = self._attachments[key]
        attachment.remove_reference()
        if attachment.collectible:
            del self._attachments[key]

    def detach_external_allocation(self, alloc, field, detach, defer):
        if defer:
            # If we need to defer this until later do that now
            self._deferred_detachments.append((alloc, field, detach))
            return
        future = self._runtime.dispatch(detach)
        # Dangle a reference to the field off the future to prevent the
        # field from being recycled until the detach is done
        future.field_reference = field
        # We also need to tell the core legate library that this array
        # is no longer attached
        self.remove_attachment(alloc)
        # If the future is already ready, then no need to track it
        if future.is_ready():
            return
        self._pending_detachments[future] = alloc

    def register_detachment(self, detach):
        key = self._next_detachment_key
        self._registered_detachments[key] = detach
        self._next_detachment_key += 1
        return key

    def remove_detachment(self, detach_key):
        detach = self._registered_detachments[detach_key]
        del self._registered_detachments[detach_key]
        return detach

    def perform_detachments(self):
        detachments = self._deferred_detachments
        self._deferred_detachments = list()
        for alloc, field, detach in detachments:
            self.detach_external_allocation(alloc, field, detach, defer=False)

    def prune_detachments(self):
        to_remove = []
        for future in self._pending_detachments.keys():
            if future.is_ready():
                to_remove.append(future)
        for future in to_remove:
            del self._pending_detachments[future]


class PartitionManager(object):
    def __init__(self, runtime):
        self._runtime = runtime
        self._num_pieces = runtime.core_context.get_tunable(
            runtime.core_library.LEGATE_CORE_TUNABLE_NUM_PIECES,
            ty.int32,
        )
        self._min_shard_volume = runtime.core_context.get_tunable(
            runtime.core_library.LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME,
            ty.int32,
        )

        self._launch_spaces = {}
        factors = list()
        pieces = self._num_pieces
        while pieces % 2 == 0:
            factors.append(2)
            pieces = pieces // 2
        while pieces % 3 == 0:
            factors.append(3)
            pieces = pieces // 3
        while pieces % 5 == 0:
            factors.append(5)
            pieces = pieces // 5
        while pieces % 7 == 0:
            factors.append(7)
            pieces = pieces // 7
        while pieces % 11 == 0:
            factors.append(11)
            pieces = pieces // 11
        if pieces > 1:
            raise ValueError(
                "legate.numpy currently doesn't support processor "
                + "counts with large prime factors greater than 11"
            )
        self._piece_factors = list(reversed(factors))

    def compute_launch_shape(self, store, restrictions):
        shape = store.shape
        assert len(restrictions) == shape.ndim

        to_partition = ()
        for dim, restriction in enumerate(restrictions):
            if restriction != Restriction.RESTRICTED:
                to_partition += (shape[dim],)

        launch_shape = self._compute_launch_shape(to_partition)
        if launch_shape is None:
            return None

        idx = 0
        result = ()
        for restriction in restrictions:
            if restriction != Restriction.RESTRICTED:
                result += (launch_shape[idx],)
                idx += 1
            else:
                result += (1,)

        return Shape(result)

    def _compute_launch_shape(self, shape):
        assert self._num_pieces > 0
        # Easy case if we only have one piece: no parallel launch space
        if self._num_pieces == 1:
            return None
        # If there is only one point or no points then we never do a parallel
        # launch
        # If we only have one point then we never do parallel launches
        elif all(ext <= 1 for ext in shape):
            return None
        # Check to see if we already did the math
        elif shape in self._launch_spaces:
            return self._launch_spaces[shape]
        # Prune out any dimensions that are 1
        temp_shape = ()
        temp_dims = ()
        volume = 1
        for dim in range(len(shape)):
            assert shape[dim] > 0
            if shape[dim] == 1:
                continue
            temp_shape = temp_shape + (shape[dim],)
            temp_dims = temp_dims + (dim,)
            volume *= shape[dim]
        # Figure out how many shards we can make with this array
        max_pieces = (
            volume + self._min_shard_volume - 1
        ) // self._min_shard_volume
        assert max_pieces > 0
        # If we can only make one piece return that now
        if max_pieces == 1:
            self._launch_spaces[shape] = None
            return None
        else:
            # TODO: a better heuristic here For now if we can make at least two
            # pieces then we will make N pieces
            max_pieces = self._num_pieces
        # Otherwise we need to compute it ourselves
        # First compute the N-th root of the number of pieces
        dims = len(temp_shape)
        temp_result = ()
        if dims == 0:
            # Project back onto the original number of dimensions
            result = ()
            for dim in range(len(shape)):
                result = result + (1,)
            return result
        elif dims == 1:
            # Easy case for one dimensional things
            temp_result = (min(temp_shape[0], max_pieces),)
        elif dims == 2:
            if volume < max_pieces:
                # TBD: Once the max_pieces heuristic is fixed, this should
                # never happen
                temp_result = temp_shape
            else:
                # Two dimensional so we can use square root to try and generate
                # as square a pieces as possible since most often we will be
                # doing matrix operations with these
                nx = temp_shape[0]
                ny = temp_shape[1]
                swap = nx > ny
                if swap:
                    temp = nx
                    nx = ny
                    ny = temp
                n = math.sqrt(float(max_pieces * nx) / float(ny))
                # Need to constraint n to be an integer with numpcs % n == 0
                # try rounding n both up and down
                n1 = int(math.floor(n + 1e-12))
                n1 = max(n1, 1)
                while max_pieces % n1 != 0:
                    n1 -= 1
                n2 = int(math.ceil(n - 1e-12))
                while max_pieces % n2 != 0:
                    n2 += 1
                # pick whichever of n1 and n2 gives blocks closest to square
                # i.e. gives the shortest long side
                side1 = max(nx // n1, ny // (max_pieces // n1))
                side2 = max(nx // n2, ny // (max_pieces // n2))
                px = n1 if side1 <= side2 else n2
                py = max_pieces // px
                # we need to trim launch space if it is larger than the
                # original shape in one of the dimensions (can happen in
                # testing)
                if swap:
                    temp_result = (
                        min(py, temp_shape[0]),
                        min(px, temp_shape[1]),
                    )
                else:
                    temp_result = (
                        min(px, temp_shape[0]),
                        min(py, temp_shape[1]),
                    )
        else:
            # For higher dimensions we care less about "square"-ness
            # and more about evenly dividing things, compute the prime
            # factors for our number of pieces and then round-robin
            # them onto the shape, with the goal being to keep the
            # last dimension >= 32 for good memory performance on the GPU
            temp_result = list()
            for dim in range(dims):
                temp_result.append(1)
            factor_prod = 1
            for factor in self._piece_factors:
                # Avoid exceeding the maximum number of pieces
                if factor * factor_prod > max_pieces:
                    break
                factor_prod *= factor
                remaining = tuple(
                    map(lambda s, r: (s + r - 1) // r, temp_shape, temp_result)
                )
                big_dim = remaining.index(max(remaining))
                if big_dim < len(temp_dims) - 1:
                    # Not the last dimension, so do it
                    temp_result[big_dim] *= factor
                else:
                    # Last dim so see if it still bigger than 32
                    if (
                        len(remaining) == 1
                        or remaining[big_dim] // factor >= 32
                    ):
                        # go ahead and do it
                        temp_result[big_dim] *= factor
                    else:
                        # Won't be see if we can do it with one of the other
                        # dimensions
                        big_dim = remaining.index(
                            max(remaining[0 : len(remaining) - 1])
                        )
                        if remaining[big_dim] // factor > 0:
                            temp_result[big_dim] *= factor
                        else:
                            # Fine just do it on the last dimension
                            temp_result[len(temp_dims) - 1] *= factor
        # Project back onto the original number of dimensions
        assert len(temp_result) == dims
        result = ()
        for dim in range(len(shape)):
            if dim in temp_dims:
                result = result + (temp_result[temp_dims.index(dim)],)
            else:
                result = result + (1,)
        result = Shape(result)
        # Save the result for later
        self._launch_spaces[shape] = result
        return result

    def compute_tile_shape(self, shape, launch_space):
        assert len(shape) == len(launch_space)
        # Over approximate the tiles so that the ends might be small
        return Shape(
            tuple(map(lambda x, y: (x + y - 1) // y, shape, launch_space))
        )

    def use_complete_tiling(self, shape, tile_shape):
        # If it would generate a very large number of elements then
        # we'll apply a heuristic for now and not actually tile it
        # TODO: A better heurisitc for this in the future
        num_tiles = (shape // tile_shape).volume()
        return not (num_tiles > 256 and num_tiles > 16 * self._num_pieces)


class Runtime(object):
    def __init__(self, core_library):
        """
        This is a class that implements the Legate runtime.
        The Runtime object provides high-level APIs for Legate libraries
        to use services in the Legion runtime. The Runtime centralizes
        resource management for all the libraries so that they can
        focus on implementing their domain logic.
        """

        try:
            self._legion_context = top_level.context[0]
        except AttributeError:
            pass

        # Record whether we need to run finalize tasks
        # Key off whether we are being loaded in a context or not
        try:
            # Do this first to detect if we're not in the top-level task
            self._legion_context = top_level.context[0]
            self._legion_runtime = legion.legion_runtime_get_runtime()
            legate_task_preamble(self._legion_runtime, self._legion_context)
            self._finalize_tasks = True
        except AttributeError:
            self._legion_runtime = None
            self._legion_context = None
            self._finalize_tasks = False

        # Initialize context lists for library registration
        self._contexts = {}
        self._context_list = []

        # Register the core library now as we need it for the rest of
        # the runtime initialization
        self.register_library(core_library)
        self._core_context = self._context_list[0]
        self._core_library = core_library

        # This list maintains outstanding operations from all legate libraries
        # to be dispatched. This list allows cross library introspection for
        # Legate operations.
        self._outstanding_ops = []
        self._window_size = 1

        # Now we initialize managers
        self._attachment_manager = AttachmentManager(self)
        self._partition_manager = PartitionManager(self)
        self.index_spaces = {}  # map shapes to index spaces
        self.region_managers = {}  # map from shape to region managers
        self.field_managers = {}  # map from (shape,dtype) to field managers

        self.destroyed = False
        self.max_field_reuse_size = 256
        self.max_field_reuse_frequency = 32
        self._empty_argmap = legion.legion_argument_map_create()

        self._next_projection_id = 1
        self._registered_projections = {}

    @property
    def legion_runtime(self):
        if self._legion_runtime is None:
            self._legion_runtime = legion.legion_runtime_get_runtime()
        return self._legion_runtime

    @property
    def legion_context(self):
        return self._legion_context

    @property
    def core_context(self):
        return self._core_context

    @property
    def core_library(self):
        return self._core_library._lib

    @property
    def empty_argmap(self):
        return self._empty_argmap

    @property
    def attachment_manager(self):
        return self._attachment_manager

    @property
    def partition_manager(self):
        return self._partition_manager

    def register_library(self, library):
        libname = library.get_name()
        if libname in self._contexts:
            raise RuntimeError(
                f"library {libname} has already been registered!"
            )
        # It's important that we load the library so that its constants
        # can be used for configuration.
        self.load_library(library)
        context = Context(self, library)
        self._contexts[libname] = context
        self._context_list.append(context)
        return context

    @staticmethod
    def load_library(library):
        shared_lib_path = library.get_shared_library()
        if shared_lib_path is not None:
            header = library.get_c_header()
            if header is not None:
                ffi.cdef(header)
            shared_lib = ffi.dlopen(shared_lib_path)
            library.initialize(shared_lib)
            callback_name = library.get_registration_callback()
            callback = getattr(shared_lib, callback_name)
            callback()
        else:
            library.initialize()

    def destroy(self):
        # Destroy all libraries. Note that we should do this
        # from the lastly added one to the first one
        for context in reversed(self._context_list):
            context.destroy()
        del self._contexts
        del self._context_list

        self._attachment_manager.destroy()

        # Remove references to our legion resources so they can be collected
        self.region_managers = None
        self.field_managers = None
        self.index_spaces = None

        if self._finalize_tasks:
            # Run a gc and then end the legate task
            gc.collect()
            legate_task_postamble(self.legion_runtime, self.legion_context)

        self.destroyed = True

    def dispatch(self, op, redop=None):
        if redop:
            return op.launch(self.legion_runtime, self.legion_context, redop)
        else:
            return op.launch(self.legion_runtime, self.legion_context)

    def _schedule(self, ops):
        must_be_single = any(op._scalar_output is not None for op in ops)
        partitioner = Partitioner(self, ops, must_be_single=must_be_single)
        strategy = partitioner.partition_stores()

        for op in ops:
            op.launch(strategy)

    def submit(self, op):
        self._outstanding_ops.append(op)
        if len(self._outstanding_ops) >= self._window_size:
            ops = self._outstanding_ops
            self._outstanding_ops = []
            self._schedule(ops)

    def _progress_unordered_operations(self):
        legion.legion_context_progress_unordered_operations(
            self.legion_runtime, self.legion_context
        )

    def unmap_region(self, physical_region, unordered=False):
        physical_region.unmap(
            self.legion_runtime, self.legion_context, unordered=unordered
        )

    def get_deliearize_functor(self):
        return self.core_context.get_projection_id(
            self.core_library.LEGATE_CORE_DELINEARIZE_FUNCTOR
        )

    def get_projection(self, src_ndim, dims):
        spec = (src_ndim, dims)
        if spec in self._registered_projections:
            return self._registered_projections[spec]

        tgt_ndim = len(dims)
        dims_c = ffi.new(f"int32_t[{tgt_ndim}]")
        for idx, dim in enumerate(dims):
            dims_c[idx] = dim

        proj_id = self.core_context.get_projection_id(self._next_projection_id)
        self._next_projection_id += 1
        self._registered_projections[spec] = proj_id

        self.core_library.legate_register_projection_functor(
            src_ndim,
            tgt_ndim,
            dims_c,
            proj_id,
        )

        return proj_id

    def get_transform_code(self, name):
        return getattr(
            self.core_library, f"LEGATE_CORE_TRANSFORM_{name.upper()}"
        )

    def create_future(self, data, size):
        future = Future()
        future.set_value(self.legion_runtime, data, size)
        return future

    def create_store(
        self,
        dtype,
        shape=None,
        storage=None,
        optimize_scalar=False,
    ):
        if shape is not None and not isinstance(shape, Shape):
            shape = Shape(shape)
        return Store(
            self,
            dtype,
            shape=shape,
            storage=storage,
            optimize_scalar=optimize_scalar,
        )

    def find_or_create_region_manager(self, shape, region=None):
        region_mgr = self.region_managers.get(shape)
        if shape not in self.region_managers:
            region_mgr = RegionManager(self, shape)
            self.region_managers[shape] = region_mgr
        return region_mgr

    def find_or_create_field_manager(self, shape, dtype):
        key = (shape, dtype)
        field_mgr = self.field_managers.get(key)
        if key not in self.field_managers:
            field_mgr = FieldManager(self, shape, dtype)
            self.field_managers[key] = field_mgr
        return field_mgr

    def allocate_field(self, shape, dtype):
        assert not self.destroyed
        region = None
        field_id = None
        field_mgr = self.find_or_create_field_manager(shape, dtype)
        region, field_id = field_mgr.allocate_field()
        field = Field(self, region, field_id, dtype, shape)
        return RegionField(self, region, field, shape)

    def free_field(self, region, field_id, dtype, shape):
        # Have a guard here to make sure that we don't try to
        # do this after we have been destroyed
        if self.destroyed:
            return
        # Now save it in our data structure for free fields eligible for reuse
        key = (shape, dtype)
        if self.field_managers is not None:
            self.field_managers[key].free_field(region, field_id)

    def import_output_region(self, out_region, field_id, dtype):
        region = out_region.get_region()
        shape = Shape(ispace=region.index_space)

        region_mgr = self.find_or_create_region_manager(shape)
        region_mgr.import_region(region)
        field = Field(
            self,
            region,
            field_id,
            dtype,
            shape,
            own=True,
        )

        self.find_or_create_field_manager(shape, dtype)

        return RegionField(self, region, field, shape)

    def create_output_region(self, fspace, fields):
        return OutputRegion(
            self.legion_context,
            self.legion_runtime,
            field_space=fspace,
            fields=fields,
        )

    def attach_external_allocation(self, context, alloc, shape, dtype, share):
        if not isinstance(alloc, ExternalAllocation):
            raise ValueError(
                "Only an ExternalAllocation object can be attached, but got"
                f"{alloc}"
            )
        if shape is not None and not isinstance(shape, Shape):
            shape = Shape(shape)
        return self._attachment_manager.attach_external_allocation(
            context,
            alloc,
            shape,
            dtype,
            share,
        )

    def has_attachment(self, array):
        return self._attachment_manager.has_attachment(array)

    def find_or_create_index_space(self, bounds):
        if bounds in self.index_spaces:
            return self.index_spaces[bounds]
        # Haven't seen this before so make it now
        rect = Rect(bounds)
        handle = legion.legion_index_space_create_domain(
            self.legion_runtime, self.legion_context, rect.raw()
        )
        result = IndexSpace(
            self.legion_context, self.legion_runtime, handle=handle
        )
        # Save this for the future
        self.index_spaces[bounds] = result
        return result

    def create_field_space(self):
        return FieldSpace(self.legion_context, self.legion_runtime)

    def create_region(self, index_space, field_space):
        handle = legion.legion_logical_region_create(
            self.legion_runtime,
            self.legion_context,
            index_space.handle,
            field_space.handle,
            True,
        )
        return Region(
            self.legion_context,
            self.legion_runtime,
            index_space,
            field_space,
            handle,
        )


_runtime = Runtime(CoreLib())


def _cleanup_legate_runtime():
    global _runtime
    _runtime.destroy()
    del _runtime
    gc.collect()


cleanup_items.append(_cleanup_legate_runtime)


def get_legion_runtime():
    return _runtime.legion_runtime


def get_legion_context():
    return _runtime.legion_context


def legate_add_library(library):
    _runtime.register_library(library)


def get_legate_runtime():
    return _runtime
