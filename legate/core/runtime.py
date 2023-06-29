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

import gc
import inspect
import math
import struct
import sys
import weakref
from collections import deque
from dataclasses import dataclass
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

from legion_top import add_cleanup_item, top_level

from ..settings import settings
from . import ffi  # Make sure we only have one ffi instance
from . import (
    Fence,
    FieldSpace,
    Future,
    FutureMap,
    IndexSpace,
    OutputRegion,
    Rect,
    Region,
    legate_task_postamble,
    legate_task_preamble,
    legion,
    types as ty,
)
from ._legion.env import LEGATE_MAX_FIELDS
from ._legion.util import Dispatchable
from .allocation import Attachable
from .communicator import CPUCommunicator, NCCLCommunicator
from .corelib import core_library
from .cycle_detector import find_cycles
from .exception import PendingException
from .machine import Machine, ProcessorKind
from .projection import is_identity_projection, pack_symbolic_projection_repr
from .restriction import Restriction
from .shape import Shape
from .utils import dlopen_no_autoclose

if TYPE_CHECKING:
    from . import ArgumentMap, Detach, IndexDetach, IndexPartition, Library
    from ._legion import (
        FieldListLike,
        Partition as LegionPartition,
        PhysicalRegion,
    )
    from .communicator import Communicator
    from .context import Context
    from .corelib import CoreLib
    from .operation import AutoTask, Copy, ManualTask, Operation
    from .partition import PartitionBase
    from .projection import SymbolicPoint
    from .store import RegionField, Store

    ProjSpec = Tuple[int, SymbolicPoint]
    ShardSpec = Tuple[int, tuple[int, int], int, int]

from math import prod

T = TypeVar("T")


_sizeof_int = ffi.sizeof("int")
_sizeof_size_t = ffi.sizeof("size_t")
assert _sizeof_size_t == 4 or _sizeof_size_t == 8

_LEGATE_FIELD_ID_BASE = 1000


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


def caller_frameinfo() -> str:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return "<unknown>"
    frame = frame.f_back.f_back
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


class LibraryAnnotations:
    def __init__(self) -> None:
        self._entries: dict[str, str] = {}
        self._provenance: Union[str, None] = None

    @property
    def provenance(self) -> Optional[str]:
        return self._provenance

    def set_provenance(self, provenance: str) -> None:
        self._provenance = provenance

    def reset_provenance(self) -> None:
        self._provenance = None

    def update(self, **kwargs: Any) -> None:
        self._entries.update(**kwargs)

    def remove(self, key: str) -> None:
        del self._entries[key]

    def __repr__(self) -> str:
        pairs = [f"{key},{value}" for key, value in self._entries.items()]
        if self._provenance is not None:
            pairs.append(f"Provenance,{self._provenance}")
        return "|".join(pairs)


# A helper class for doing field management with control replication
@dataclass(frozen=True)
class FreeFieldInfo:
    manager: FieldManager
    region: Region
    field_id: int

    def free(self, ordered: bool = False) -> None:
        self.manager.free_field(self.region, self.field_id, ordered=ordered)


class FieldMatch(Dispatchable[Future]):
    __slots__ = ["manager", "fields", "input", "output", "future"]

    def __init__(self, fields: List[FreeFieldInfo]) -> None:
        self.fields = fields
        # Allocate arrays of ints that are twice as long as fields since
        # our values will be 'field_id,tree_id' pairs
        if (num_fields := len(fields)) > 0:
            alloc_string = f"int[{2 * num_fields}]"
            self.input = ffi.new(alloc_string)
            self.output = ffi.new(alloc_string)
            # Fill in the input buffer with our data
            for idx in range(num_fields):
                field = fields[idx]
                self.input[2 * idx] = field.region.handle.tree_id
                self.input[2 * idx + 1] = field.field_id
        else:
            self.input = ffi.NULL
            self.output = ffi.NULL
        self.future: Union[Future, None] = None

    def launch(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        **kwargs: Any,
    ) -> Future:
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

    def update_free_fields(self) -> None:
        # If we know there are no fields then we can be done early
        if len(self.fields) == 0:
            return

        assert self.future is not None

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
            ordered_fields: List[Optional[FreeFieldInfo]] = [None] * num_fields
            for field in self.fields:
                found = False
                for idx in range(num_fields):
                    if self.output[2 * idx] != field.region.handle.tree_id:
                        continue
                    if self.output[2 * idx + 1] != field.field_id:
                        continue
                    assert ordered_fields[idx] is None
                    ordered_fields[idx] = field
                    found = True
                    break
                if not found:
                    # Not found so put it back int the unordered queue
                    field.free(ordered=False)
            # Notice that we do this in the order of the list which is the
            # same order as they were in the array returned by the match
            fields = (field for field in ordered_fields if field is not None)
            for field in fields:
                field.free(ordered=True)
        else:
            # No fields on all shards so put all our fields back into
            # the unorered queue
            for field in self.fields:
                field.free(ordered=False)


# A simple manager that keeps track of free fields from all free managers
# and outstanding field matches issued for them.
class FieldMatchManager:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._freed_fields: List[FreeFieldInfo] = []
        self._matches: Deque[FieldMatch] = deque()
        self._match_counter = 0
        self._match_frequency = runtime.max_field_reuse_frequency

    def add_free_field(
        self, manager: FieldManager, region: Region, field_id: int
    ) -> None:
        self._freed_fields.append(FreeFieldInfo(manager, region, field_id))

    def issue_field_match(self, credit: int) -> None:
        # Increment our match counter
        self._match_counter += credit
        if self._match_counter < self._match_frequency:
            return
        # If the match counter equals our match frequency then do an exchange

        # This is where the rubber meets the road between control
        # replication and garbage collection. We need to see if there
        # are any freed fields that are shared across all the shards.
        # We have to test this deterministically no matter what even
        # if we don't have any fields to offer ourselves because this
        # is a collective with other shards. If we have any we can use
        # the first one and put the remainder on our free fields list
        # so that we can reuse them later. If there aren't any then
        # all the shards will go allocate a new field.
        local_free_fields = self._freed_fields
        # The match now owns our freed fields so make a new list
        # Have to do this before dispatching the match
        self._freed_fields = []
        match = FieldMatch(local_free_fields)
        # Dispatch the match. Note that this is necessary even when
        # the field list is empty, as other shards might have non-empty
        # lists and all shards should participate the match.
        self._runtime.dispatch(match)
        # Put it on the deque of outstanding matches
        self._matches.append(match)
        # Reset the match counter back to 0
        self._match_counter = 0

    def update_free_fields(self) -> None:
        while len(self._matches) > 0:
            match = self._matches.popleft()
            match.update_free_fields()


# This class keeps track of usage of a single region
class RegionManager:
    def __init__(
        self, shape: Shape, region: Region, imported: bool = False
    ) -> None:
        self._shape = shape
        self._region = region
        # Monotonically increases as more fields are allocated
        self._alloc_field_count = 0
        # Fluctuates based on field usage
        self._active_field_count = 0
        self._next_field_id = _LEGATE_FIELD_ID_BASE
        self._imported = imported

    @property
    def region(self) -> Region:
        return self._region

    @property
    def shape(self) -> Shape:
        return self._shape

    def destroy(self, unordered: bool) -> None:
        # An explicit destruction has a benefit that we can sometimes perform
        # ordered destructions, whereas in a destructor we can only do
        # unordered destructions
        self._region.destroy(unordered)

    def increase_active_field_count(self) -> bool:
        revived = self._active_field_count == 0
        self._active_field_count += 1
        return revived

    def decrease_active_field_count(self) -> bool:
        self._active_field_count -= 1
        return self._active_field_count == 0

    def increase_field_count(self) -> bool:
        fresh = self._alloc_field_count == 0
        self._alloc_field_count += 1
        revived = self.increase_active_field_count()
        return not fresh and revived

    @property
    def has_space(self) -> bool:
        return self._alloc_field_count < LEGATE_MAX_FIELDS

    def get_next_field_id(self) -> int:
        field_id = self._next_field_id
        self._next_field_id += 1
        return field_id

    def allocate_field(self, field_size: Any) -> tuple[Region, int, bool]:
        field_id = self._region.field_space.allocate_field(
            field_size, self.get_next_field_id()
        )
        revived = self.increase_field_count()
        return self._region, field_id, revived


# This class manages the allocation and reuse of fields
class FieldManager:
    def __init__(
        self, runtime: Runtime, shape: Shape, field_size: int
    ) -> None:
        assert isinstance(field_size, int)
        self.runtime = runtime
        self.shape = shape
        self.field_size = field_size
        # This is a sanitized list of (region,field_id) pairs that is
        # guaranteed to be ordered across all the shards even with
        # control replication
        self.free_fields: Deque[tuple[Region, int]] = deque()

    def destroy(self) -> None:
        self.free_fields = deque()

    def try_reuse_field(self) -> Optional[tuple[Region, int]]:
        return (
            self.free_fields.popleft() if len(self.free_fields) > 0 else None
        )

    def allocate_field(self) -> tuple[Region, int]:
        if (result := self.try_reuse_field()) is not None:
            region_manager = self.runtime.find_region_manager(result[0])
            if region_manager.increase_active_field_count():
                self.runtime.revive_manager(region_manager)
            return result
        region_manager = self.runtime.find_or_create_region_manager(self.shape)
        region, field_id, revived = region_manager.allocate_field(
            self.field_size
        )
        if revived:
            self.runtime.revive_manager(region_manager)
        return region, field_id

    def free_field(
        self, region: Region, field_id: int, ordered: bool = False
    ) -> None:
        self.free_fields.append((region, field_id))
        region_manager = self.runtime.find_region_manager(region)
        if region_manager.decrease_active_field_count():
            self.runtime.free_region_manager(
                self.shape, region, unordered=not ordered
            )

    def remove_all_fields(self, region: Region) -> None:
        self.free_fields = deque(f for f in self.free_fields if f[0] != region)


class ConsensusMatchingFieldManager(FieldManager):
    def __init__(
        self, runtime: Runtime, shape: Shape, field_size: int
    ) -> None:
        super().__init__(runtime, shape, field_size)
        self._field_match_manager = runtime.field_match_manager
        self._update_match_credit()

    def _update_match_credit(self) -> None:
        if self.shape.fixed:
            size = self.shape.volume() * self.field_size
            self._match_credit = (
                size + self.runtime.max_field_reuse_size - 1
                if size > self.runtime.max_field_reuse_size
                else self.runtime.max_field_reuse_size
            ) // self.runtime.max_field_reuse_size
            # No need to update the credit as the exact size is known
            self._need_to_update_match_credit = False
        # If the shape is unknown, we set the credit such that every new
        # free field leads to a consensus match, and ask the manager
        # to update the credit.
        else:
            self._match_credit = self.runtime.max_field_reuse_frequency
            self._need_to_update_match_credit = True

    def try_reuse_field(self) -> Optional[tuple[Region, int]]:
        if self._need_to_update_match_credit:
            self._update_match_credit()
        self._field_match_manager.issue_field_match(self._match_credit)

        # First, if we have a free field then we know everyone has one of those
        if len(self.free_fields) > 0:
            return self.free_fields.popleft()

        self._field_match_manager.update_free_fields()

        # Check again to see if we have any free fields
        return (
            self.free_fields.popleft() if len(self.free_fields) > 0 else None
        )

    def free_field(
        self, region: Region, field_id: int, ordered: bool = False
    ) -> None:
        if ordered:
            super().free_field(region, field_id, ordered=ordered)
        else:  # Put this on the unordered list
            self._field_match_manager.add_free_field(self, region, field_id)


class Attachment:
    def __init__(
        self, ptr: int, extent: int, shareable: bool, region_field: RegionField
    ) -> None:
        self.ptr = ptr
        self.extent = extent
        self.end = ptr + extent - 1
        # Catch a case where we try to (individually) re-attach a buffer that
        # was used in an IndexAttach. In that case it would be wrong to return
        # the RegionField produced by that IndexAttach, since the buffer in
        # question only covers a part of that.
        self.shareable = shareable
        self._region_field = weakref.ref(region_field)

    def overlaps(self, other: Attachment) -> bool:
        return not (self.end < other.ptr or other.end < self.ptr)

    @property
    def region_field(self) -> Optional[RegionField]:
        return self._region_field()

    @region_field.setter
    def region_field(self, region_field: RegionField) -> None:
        self._region_field = weakref.ref(region_field)


class AttachmentManager:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._attachments: dict[tuple[int, int], Attachment] = dict()
        self._next_detachment_key = 0
        self._registered_detachments: dict[
            int, Union[Detach, IndexDetach]
        ] = dict()
        self._deferred_detachments: List[
            tuple[Attachable, Union[Detach, IndexDetach]]
        ] = list()
        self._pending_detachments: dict[Future, Attachable] = dict()
        self._destroyed = False

    def destroy(self) -> None:
        self._destroyed = True
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
        self._attachments = dict()

    @staticmethod
    def attachment_key(buf: memoryview) -> tuple[int, int]:
        assert isinstance(buf, memoryview)
        if not buf.contiguous:
            raise RuntimeError("Cannot attach to non-contiguous buffer")
        ptr = ffi.cast("uintptr_t", ffi.from_buffer(buf))
        base_ptr = int(ptr)  # type: ignore[call-overload]
        return (base_ptr, buf.nbytes)

    def has_attachment(self, buf: memoryview) -> bool:
        key = self.attachment_key(buf)
        attachment = self._attachments.get(key, None)
        return attachment is not None and attachment.region_field is not None

    def reuse_existing_attachment(
        self, buf: memoryview
    ) -> Optional[RegionField]:
        key = self.attachment_key(buf)
        attachment = self._attachments.get(key, None)
        if attachment is None:
            return None
        rf = attachment.region_field
        # If the region field is already collected, we don't need to keep
        # track of it for de-duplication.
        if rf is None:
            del self._attachments[key]
            return None
        return rf if attachment.shareable else None

    def _add_attachment(
        self, buf: memoryview, shareable: bool, region_field: RegionField
    ) -> None:
        key = self.attachment_key(buf)
        attachment = self._attachments.get(key, None)
        if not (attachment is None or attachment.region_field is None):
            raise RuntimeError(
                "Cannot attach two different RegionFields to the same buffer"
            )
        # If the region field is already collected, we don't need to keep
        # track of it for de-duplication.
        if attachment is not None:
            del self._attachments[key]
        attachment = Attachment(*key, shareable, region_field)
        for other in self._attachments.values():
            if other.overlaps(attachment):
                raise RuntimeError(
                    "Aliased attachments not supported by Legate"
                )
        self._attachments[key] = attachment

    def attach_external_allocation(
        self, alloc: Attachable, region_field: RegionField
    ) -> None:
        if isinstance(alloc, memoryview):
            self._add_attachment(alloc, True, region_field)
        else:
            for buf in alloc.shard_local_buffers.values():
                self._add_attachment(buf, False, region_field)

    def _remove_attachment(self, buf: memoryview) -> None:
        # If the manager is already destroyed, ignore the detachment
        # attempt
        if self._destroyed:
            return
        key = self.attachment_key(buf)
        if key not in self._attachments:
            raise RuntimeError("Unable to find attachment to remove")
        del self._attachments[key]

    def _remove_allocation(self, alloc: Attachable) -> None:
        if isinstance(alloc, memoryview):
            self._remove_attachment(alloc)
        else:
            for buf in alloc.shard_local_buffers.values():
                self._remove_attachment(buf)

    def detach_external_allocation(
        self,
        alloc: Attachable,
        detach: Union[Detach, IndexDetach],
        defer: bool = False,
        previously_deferred: bool = False,
    ) -> None:
        # If the detachment was previously deferred, then we don't
        # need to remove the allocation from the map again.
        if not previously_deferred:
            self._remove_allocation(alloc)
        if defer:
            # If we need to defer this until later do that now
            self._deferred_detachments.append((alloc, detach))
            return
        future = self._runtime.dispatch(detach)
        # Dangle a reference to the field off the future to prevent the
        # field from being recycled until the detach is done
        field = detach.field  # type: ignore[union-attr]
        future.field_reference = field  # type: ignore[attr-defined]
        # If the future is already ready, then no need to track it
        if future.is_ready():
            return
        self._pending_detachments[future] = alloc

    def register_detachment(self, detach: Union[Detach, IndexDetach]) -> int:
        key = self._next_detachment_key
        self._registered_detachments[key] = detach
        self._next_detachment_key += 1
        return key

    def remove_detachment(self, detach_key: int) -> Union[Detach, IndexDetach]:
        detach = self._registered_detachments[detach_key]
        del self._registered_detachments[detach_key]
        return detach

    def perform_detachments(self) -> None:
        detachments = self._deferred_detachments
        self._deferred_detachments = list()
        for alloc, detach in detachments:
            self.detach_external_allocation(
                alloc, detach, defer=False, previously_deferred=True
            )

    def prune_detachments(self) -> None:
        to_remove = []
        for future in self._pending_detachments.keys():
            if future.is_ready():
                to_remove.append(future)
        for future in to_remove:
            del self._pending_detachments[future]


class PartitionManager:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._min_shard_volume = runtime.core_context.get_tunable(
            runtime.core_library.LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME,
            ty.int64,
        )

        self._launch_spaces: dict[
            tuple[int, tuple[int, ...]], Optional[tuple[int, ...]]
        ] = {}
        self._piece_factors: dict[int, list[int]] = {}

        self._index_partitions: dict[
            tuple[IndexSpace, PartitionBase], IndexPartition
        ] = {}
        # Maps storage id-partition pairs to Legion partitions
        self._legion_partitions: dict[
            tuple[int, PartitionBase], Union[None, LegionPartition]
        ] = {}
        self._storage_key_partitions: dict[tuple[int, int], PartitionBase] = {}
        self._store_key_partitions: dict[tuple[int, int], PartitionBase] = {}

    def get_current_num_pieces(self) -> int:
        return len(self._runtime.machine)

    def get_piece_factors(self) -> list[int]:
        num_pieces = self.get_current_num_pieces()
        if num_pieces in self._piece_factors:
            return self._piece_factors[num_pieces]

        factors: list[int] = []

        remaining = num_pieces

        def record_factors(remaining: int, val: int) -> int:
            while remaining % val == 0:
                factors.append(val)
                remaining = remaining // val
            return remaining

        for factor in (2, 3, 5, 7, 11):
            remaining = record_factors(remaining, factor)

        if remaining > 1:
            raise ValueError(
                "Legate currently doesn't support processor "
                "counts with large prime factors greater than 11"
            )

        factors = list(reversed(factors))
        self._piece_factors[num_pieces] = factors
        return factors

    def compute_launch_shape(
        self, store: Store, restrictions: tuple[Restriction, ...]
    ) -> Optional[Shape]:
        shape = store.shape
        assert len(restrictions) == shape.ndim

        to_partition: tuple[int, ...] = tuple(
            shape[dim]
            for dim, restriction in enumerate(restrictions)
            if restriction != Restriction.RESTRICTED
        )

        if prod(to_partition) == 0:
            return None

        launch_shape = self._compute_launch_shape(to_partition)
        if launch_shape is None:
            return None

        idx = 0
        result: tuple[int, ...] = ()
        for restriction in restrictions:
            if restriction != Restriction.RESTRICTED:
                result += (launch_shape[idx],)
                idx += 1
            else:
                result += (1,)

        return Shape(result)

    def _compute_launch_shape(
        self, shape: tuple[int, ...]
    ) -> Optional[tuple[int, ...]]:
        num_pieces = self.get_current_num_pieces()
        key = (num_pieces, shape)
        # Easy case if we only have one piece: no parallel launch space
        if num_pieces == 1:
            return None
        # If there is only one point or no points then we never do a parallel
        # launch
        # If we only have one point then we never do parallel launches
        elif all(ext <= 1 for ext in shape):
            return None
        # Check to see if we already did the math
        elif (match := self._launch_spaces.get(key)) is not None:
            return match
        # Prune out any dimensions that are 1
        temp_shape: tuple[int, ...] = ()
        temp_dims: tuple[int, ...] = ()
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
            self._launch_spaces[key] = None
            return None
        else:
            # TODO: a better heuristic here For now if we can make at least two
            # pieces then we will make N pieces
            max_pieces = num_pieces
        # Otherwise we need to compute it ourselves
        # First compute the N-th root of the number of pieces
        dims = len(temp_shape)
        temp_result: tuple[int, ...] = ()
        if dims == 0:
            # Project back onto the original number of dimensions
            result: tuple[int, ...] = ()
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
            local_result: List[int] = [1] * dims
            factor_prod = 1
            piece_factors = self.get_piece_factors()
            for factor in piece_factors:
                # Avoid exceeding the maximum number of pieces
                if factor * factor_prod > max_pieces:
                    break
                factor_prod *= factor
                remaining = tuple(
                    map(
                        lambda s, r: (s + r - 1) // r, temp_shape, local_result
                    )
                )
                big_dim = remaining.index(max(remaining))
                if big_dim < len(temp_dims) - 1:
                    # Not the last dimension, so do it
                    local_result[big_dim] *= factor
                else:
                    # Last dim so see if it still bigger than 32
                    if (
                        len(remaining) == 1
                        or remaining[big_dim] // factor >= 32
                    ):
                        # go ahead and do it
                        local_result[big_dim] *= factor
                    else:
                        # Won't be see if we can do it with one of the other
                        # dimensions
                        big_dim = remaining.index(
                            max(remaining[0 : len(remaining) - 1])
                        )
                        if remaining[big_dim] // factor > 0:
                            local_result[big_dim] *= factor
                        else:
                            # Fine just do it on the last dimension
                            local_result[len(temp_dims) - 1] *= factor
            temp_result = tuple(local_result)
        # Project back onto the original number of dimensions
        assert len(temp_result) == dims
        result = ()
        for dim in range(len(shape)):
            if dim in temp_dims:
                result = result + (temp_result[temp_dims.index(dim)],)
            else:
                result = result + (1,)
        # Save the result for later
        self._launch_spaces[key] = result
        return result

    def compute_tile_shape(self, shape: Shape, launch_space: Shape) -> Shape:
        assert len(shape) == len(launch_space)
        # Over approximate the tiles so that the ends might be small
        return Shape(
            tuple(map(lambda x, y: (x + y - 1) // y, shape, launch_space))
        )

    def use_complete_tiling(self, shape: Shape, tile_shape: Shape) -> bool:
        # If it would generate a very large number of elements then
        # we'll apply a heuristic for now and not actually tile it
        # TODO: A better heuristic for this in the future
        num_tiles = (shape // tile_shape).volume()
        num_pieces = self.get_current_num_pieces()
        return not (num_tiles > 256 and num_tiles > 16 * num_pieces)

    def find_index_partition(
        self, index_space: IndexSpace, functor: PartitionBase
    ) -> Union[IndexPartition, None]:
        key = (index_space, functor)
        return self._index_partitions.get(key)

    def record_index_partition(
        self,
        functor: PartitionBase,
        index_partition: IndexPartition,
    ) -> None:
        key = (index_partition.parent, functor)
        assert key not in self._index_partitions
        self._index_partitions[key] = index_partition

    def find_store_key_partition(
        self, store_id: int, restrictions: tuple[Restriction, ...]
    ) -> Union[None, PartitionBase]:
        key = (self.get_current_num_pieces(), store_id)
        partition = self._store_key_partitions.get(key)
        if partition is not None and not partition.satisfies_restriction(
            restrictions
        ):
            partition = None
        return partition

    def record_store_key_partition(
        self, store_id: int, key_partition: PartitionBase
    ) -> None:
        key = (self.get_current_num_pieces(), store_id)
        self._store_key_partitions[key] = key_partition

    def reset_store_key_partition(self, store_id: int) -> None:
        key = (self.get_current_num_pieces(), store_id)
        if key in self._store_key_partitions:
            del self._store_key_partitions[key]

    def find_storage_key_partition(
        self, storage_id: int, restrictions: tuple[Restriction, ...]
    ) -> Union[None, PartitionBase]:
        key = (self.get_current_num_pieces(), storage_id)
        partition = self._storage_key_partitions.get(key)
        if partition is not None and not partition.satisfies_restriction(
            restrictions
        ):
            partition = None
        return partition

    def record_storage_key_partition(
        self, storage_id: int, key_partition: PartitionBase
    ) -> None:
        key = (self.get_current_num_pieces(), storage_id)
        self._storage_key_partitions[key] = key_partition

    def reset_storage_key_partition(self, storage_id: int) -> None:
        key = (self.get_current_num_pieces(), storage_id)
        if key in self._storage_key_partitions:
            del self._storage_key_partitions[key]

    def find_legion_partition(
        self, storage_id: int, functor: PartitionBase
    ) -> tuple[Optional[LegionPartition], bool]:
        key = (storage_id, functor)
        found = key in self._legion_partitions
        part = self._legion_partitions.get(key)
        return part, found

    def record_legion_partition(
        self,
        storage_id: int,
        functor: PartitionBase,
        legion_partition: Optional[LegionPartition],
    ) -> None:
        key = (storage_id, functor)
        self._legion_partitions[key] = legion_partition


class CommunicatorManager:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._nccl = NCCLCommunicator(runtime)

        self._cpu = (
            None if settings.disable_mpi() else CPUCommunicator(runtime)
        )

    def destroy(self) -> None:
        self._nccl.destroy()
        if self._cpu:
            self._cpu.destroy()

    def get_nccl_communicator(self) -> Communicator:
        return self._nccl

    def get_cpu_communicator(self) -> Communicator:
        if self._cpu is None:
            raise RuntimeError("MPI is disabled")
        return self._cpu

    @property
    def has_cpu_communicator(self) -> bool:
        return self._cpu is not None


class Runtime:
    _legion_runtime: Union[legion.legion_runtime_t, None]
    _legion_context: Union[legion.legion_context_t, None]

    def __init__(self, core_library: CoreLib) -> None:
        """
        This is a class that implements the Legate runtime.
        The Runtime object provides high-level APIs for Legate libraries
        to use services in the Legion runtime. The Runtime centralizes
        resource management for all the libraries so that they can
        focus on implementing their domain logic.
        """

        # Record whether we need to run finalize tasks
        # Key off whether we are being loaded in a context or not
        try:
            # Do this first to detect if we're not in the top-level task
            self._legion_context = top_level.context[0]
            self._legion_runtime = legion.legion_runtime_get_runtime()
            assert self._legion_runtime is not None
            assert self._legion_context is not None
            legate_task_preamble(self._legion_runtime, self._legion_context)
            self._finalize_tasks = True
        except AttributeError:
            self._legion_runtime = None
            self._legion_context = None
            self._finalize_tasks = False

        # Initialize context lists for library registration
        self._contexts: dict[str, Context] = {}
        self._context_list: List[Context] = []

        # Register the core library now as we need it for the rest of
        # the runtime initialization
        self.register_library(core_library)
        self._core_context = self._context_list[0]
        self._core_library = core_library

        self._unique_op_id = 0
        # This list maintains outstanding operations from all legate libraries
        # to be dispatched. This list allows cross library introspection for
        # Legate operations.
        self._outstanding_ops: List[Operation] = []
        self._window_size = self._core_context.get_tunable(
            legion.LEGATE_CORE_TUNABLE_WINDOW_SIZE,
            ty.uint32,
        )

        self._next_store_id = 0
        self._next_storage_id = 0

        self._barriers: List[legion.legion_phase_barrier_t] = []
        self.nccl_needs_barrier = bool(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_NCCL_NEEDS_BARRIER,
                ty.bool_,
            )
        )

        self._num_nodes = int(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_NUM_NODES,
                ty.int32,
            )
        )
        self.max_field_reuse_frequency = int(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY,
                ty.uint32,
            )
        )
        self.max_field_reuse_size = int(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE,
                ty.uint64,
            )
        )
        self._field_manager_class = (
            ConsensusMatchingFieldManager
            if self._num_nodes > 1 or settings.consensus()
            else FieldManager
        )
        self._max_lru_length = int(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_MAX_LRU_LENGTH,
                ty.uint32,
            )
        )

        # Now we initialize managers
        self._machines = [Machine.create_toplevel_machine(self)]
        self._attachment_manager = AttachmentManager(self)
        self._partition_manager = PartitionManager(self)
        self._comm_manager = CommunicatorManager(self)
        self._field_match_manager = FieldMatchManager(self)
        # map shapes to index spaces
        self.index_spaces: dict[Rect, IndexSpace] = {}
        # map from shapes to active region managers
        self.active_region_managers: dict[Shape, RegionManager] = {}
        # map from regions to their managers
        self.region_managers_by_region: dict[Region, RegionManager] = {}
        # LRU for free region managers
        self.lru_managers: Deque[RegionManager] = deque()
        # map from (shape,dtype) to field managers
        self.field_managers: dict[tuple[Shape, Any], FieldManager] = {}

        self.destroyed = False
        self._empty_argmap: ArgumentMap = legion.legion_argument_map_create()

        # A projection functor and its corresponding sharding functor
        # have the same local id
        first_functor_id: int = (
            core_library._lib.LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID  # type: ignore[union-attr] # noqa: E501
        )
        self._next_projection_id = first_functor_id
        self._next_sharding_id = first_functor_id
        self._registered_projections: dict[ProjSpec, int] = {}
        self._registered_shardings: dict[ShardSpec, int] = {}

        self._max_pending_exceptions: int = int(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_MAX_PENDING_EXCEPTIONS,
                ty.uint32,
            )
        )
        self._precise_exception_trace: bool = bool(
            self._core_context.get_tunable(
                legion.LEGATE_CORE_TUNABLE_PRECISE_EXCEPTION_TRACE,
                ty.bool_,
            )
        )

        self._pending_exceptions: list[PendingException] = []
        self._annotations: list[LibraryAnnotations] = [LibraryAnnotations()]

        # TODO: We can make this a true loading-time constant with Cython
        self._variant_ids = {
            ProcessorKind.GPU: self.core_library.LEGATE_GPU_VARIANT,
            ProcessorKind.OMP: self.core_library.LEGATE_OMP_VARIANT,
            ProcessorKind.CPU: self.core_library.LEGATE_CPU_VARIANT,
        }

        self._array_type_cache: dict[tuple[ty.Dtype, int], ty.Dtype] = {}

    @property
    def has_cpu_communicator(self) -> bool:
        return self._comm_manager.has_cpu_communicator

    @property
    def legion_runtime(self) -> legion.legion_runtime_t:
        if self._legion_runtime is None:
            self._legion_runtime = legion.legion_runtime_get_runtime()
        return self._legion_runtime

    @property
    def legion_context(self) -> legion.legion_context_t:
        if self._legion_context is None:
            self._legion_context = top_level.context[0]
            assert self._legion_context is not None
        return self._legion_context

    @property
    def core_context(self) -> Context:
        return self._core_context

    @property
    def core_library(self) -> Any:
        return self._core_library._lib

    @property
    def empty_argmap(self) -> ArgumentMap:
        return self._empty_argmap

    @property
    def variant_ids(self) -> dict[ProcessorKind, int]:
        return self._variant_ids

    @property
    def machine(self) -> Machine:
        """
        Returns the machine object for the current scope

        Returns
        -------
        Machine
            Machine object
        """
        return self._machines[-1]

    def push_machine(self, machine: Machine) -> None:
        self._machines.append(machine)

    def pop_machine(self) -> None:
        if len(self._machines) <= 1:
            raise RuntimeError("Machine stack underflow")
        self._machines.pop()

    @property
    def attachment_manager(self) -> AttachmentManager:
        return self._attachment_manager

    @property
    def partition_manager(self) -> PartitionManager:
        return self._partition_manager

    @property
    def field_match_manager(self) -> FieldMatchManager:
        return self._field_match_manager

    @property
    def annotation(self) -> LibraryAnnotations:
        """
        Returns the current set of annotations. Provenance string is one
        entry in the set.

        Returns
        -------
        LibraryAnnotations
            Library annotations
        """
        return self._annotations[-1]

    def get_all_annotations(self) -> str:
        return str(self.annotation)

    @property
    def provenance(self) -> Optional[str]:
        """
        Returns the current provenance string. Attached to every operation
        issued with the context.

        Returns
        -------
        str or None
            Provenance string
        """
        return self.annotation.provenance

    def register_library(self, library: Library) -> Context:
        """
        Registers a library to the runtime.

        Parameters
        ----------
        library : Library
            Library object

        Returns
        -------
        Context
            A new context for the library
        """
        from .context import Context

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
    def load_library(library: Library) -> None:
        shared_lib_path = library.get_shared_library()
        if shared_lib_path is not None:
            header = library.get_c_header()
            if header is not None:
                ffi.cdef(header)
            # Don't use ffi.dlopen(), because that will call dlclose()
            # automatically when the object gets collected, thus removing
            # symbols that may be needed when destroying C++ objects later
            # (e.g. vtable entries, which will be queried for virtual
            # destructors), causing errors at shutdown.
            shared_lib = dlopen_no_autoclose(ffi, shared_lib_path)
            library.initialize(shared_lib)
            callback_name = library.get_registration_callback()
            callback = getattr(shared_lib, callback_name)
            callback()
        else:
            library.initialize(None)

    def destroy(self) -> None:
        # Before we clean up the runtime, we should execute all outstanding
        # operations.
        self.flush_scheduling_window()

        # Then we also need to raise all exceptions if there were any
        self.raise_exceptions()

        self._comm_manager.destroy()
        for barrier in self._barriers:
            legion.legion_phase_barrier_destroy(
                self.legion_runtime, self.legion_context, barrier
            )

        # Destroy all libraries. Note that we should do this
        # from the lastly added one to the first one
        for context in reversed(self._context_list):
            context.destroy()
        del self._contexts
        del self._context_list

        self._attachment_manager.destroy()

        # Remove references to our legion resources so they can be collected
        self.active_region_managers = {}
        self.region_managers_by_region = {}
        self.field_managers = {}
        self.index_spaces = {}
        # Explicitly release the reference to the partition manager so that
        # it may be collected, releasing references to Futures and FutureMaps.
        del self._partition_manager

        if self._finalize_tasks:
            # Run a gc and then end the legate task
            gc.collect()
            legate_task_postamble(self.legion_runtime, self.legion_context)

        self.destroyed = True

    def get_unique_op_id(self) -> int:
        op_id = self._unique_op_id
        self._unique_op_id += 1
        return op_id

    def get_next_store_id(self) -> int:
        self._next_store_id += 1
        return self._next_store_id

    def get_next_storage_id(self) -> int:
        self._next_storage_id += 1
        return self._next_storage_id

    def dispatch(self, op: Dispatchable[T]) -> T:
        self._attachment_manager.perform_detachments()
        self._attachment_manager.prune_detachments()
        return op.launch(self.legion_runtime, self.legion_context)

    def dispatch_single(self, op: Dispatchable[T]) -> T:
        self._attachment_manager.perform_detachments()
        self._attachment_manager.prune_detachments()
        return op.launch(self.legion_runtime, self.legion_context)

    def _schedule(self, ops: List[Operation]) -> None:
        from .solver import Partitioner

        # TODO: For now we run the partitioner for each operation separately.
        #       We will eventually want to compute a trace-wide partitioning
        #       strategy.
        strategies = []
        for op in ops:
            must_be_single = len(op.scalar_outputs) > 0
            partitioner = Partitioner([op], must_be_single=must_be_single)
            # TODO: When we start partitioning a batch of operations, changes
            # of machine configuration would delineat the batches
            with op.target_machine:
                strategies.append(partitioner.partition_stores())

        for op, strategy in zip(ops, strategies):
            with op.target_machine:
                op.launch(strategy)

    def flush_scheduling_window(self) -> None:
        if len(self._outstanding_ops) == 0:
            return
        ops = self._outstanding_ops
        self._outstanding_ops = []
        self._schedule(ops)

    def submit(self, op: Operation) -> None:
        if op.can_raise_exception and self._precise_exception_trace:
            op.capture_traceback()
        self._outstanding_ops.append(op)
        if len(self._outstanding_ops) >= self._window_size:
            self.flush_scheduling_window()
        if len(self._pending_exceptions) >= self._max_pending_exceptions:
            self.raise_exceptions()

    def _progress_unordered_operations(self) -> None:
        legion.legion_context_progress_unordered_operations(
            self.legion_runtime, self.legion_context
        )

    def unmap_region(
        self, physical_region: PhysicalRegion, unordered: bool = False
    ) -> None:
        physical_region.unmap(
            self.legion_runtime, self.legion_context, unordered=unordered
        )

    def get_delinearize_functor(self) -> int:
        return self.core_context.get_projection_id(
            self.core_library.LEGATE_CORE_DELINEARIZE_PROJ_ID
        )

    def _register_projection_functor(
        self,
        spec: ProjSpec,
        src_ndim: int,
        tgt_ndim: int,
        dims_c: Any,
        weights_c: Any,
        offsets_c: Any,
    ) -> int:
        proj_id = self.core_context.get_projection_id(self._next_projection_id)
        self._next_projection_id += 1
        self._registered_projections[spec] = proj_id

        self.core_library.legate_register_affine_projection_functor(
            src_ndim,
            tgt_ndim,
            dims_c,
            weights_c,
            offsets_c,
            proj_id,
        )

        return proj_id

    def get_sharding(self, proj_id: int) -> int:
        proc_range = self.machine.get_processor_range()
        offset = proc_range.low % proc_range.per_node_count
        node_range = self.machine.get_node_range()
        shard_spec = (proj_id, node_range, offset, proc_range.per_node_count)

        if shard_spec in self._registered_shardings:
            return self._registered_shardings[shard_spec]

        shard_id = self.core_context.get_sharding_id(self._next_sharding_id)
        self._next_sharding_id += 1
        self._registered_shardings[shard_spec] = shard_id

        self.core_library.legate_create_sharding_functor_using_projection(
            shard_id,
            proj_id,
            node_range[0],
            node_range[1],
            offset,
            proc_range.per_node_count,
        )

        self._registered_shardings[shard_spec] = shard_id

        return shard_id

    def get_projection(self, src_ndim: int, dims: SymbolicPoint) -> int:
        proj_spec = (src_ndim, dims)

        proj_id: int
        if proj_spec in self._registered_projections:
            proj_id = self._registered_projections[proj_spec]
        else:
            if is_identity_projection(src_ndim, dims):
                proj_id = 0
            else:
                proj_id = self._register_projection_functor(
                    proj_spec, *pack_symbolic_projection_repr(src_ndim, dims)
                )
            self._registered_projections[proj_spec] = proj_id

        return proj_id

    def get_transform_code(self, name: str) -> int:
        return getattr(
            self.core_library, f"LEGATE_CORE_TRANSFORM_{name.upper()}"
        )

    def create_future(self, data: Any, size: int) -> Future:
        """
        Creates a future from a buffer holding a scalar value. The value is
        copied to the future.

        Parameters
        ----------
        data : buffer
            Buffer that holds a scalar value

        size : int
            Size of the value

        Returns
        -------
        Future
            A new future
        """
        future = Future()
        future.set_value(self.legion_runtime, data, size)
        return future

    def create_store(
        self,
        dtype: ty.Dtype,
        shape: Optional[Union[Shape, tuple[int, ...]]] = None,
        data: Optional[Union[RegionField, Future]] = None,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> Store:
        from .store import RegionField, Storage, Store

        if not isinstance(dtype, ty.Dtype):
            raise ValueError(f"Unsupported type: {dtype}")

        if ndim is not None and shape is not None:
            raise ValueError("ndim cannot be used with shape")
        elif ndim is None and shape is None:
            ndim = 1

        if shape is not None and not isinstance(shape, Shape):
            shape = Shape(shape)

        kind = (
            Future
            if optimize_scalar and shape is not None and shape.volume() == 1
            else RegionField
        )

        sanitized_shape: Optional[Shape]
        if kind is RegionField and shape is not None and shape.ndim == 0:
            from .transform import Project, identity

            # If the client requested a 0D region-backed store, we need to
            # promote the shape to 1D to create the storage, as Legion
            # doesn't allow 0D regions. And we also need to set up a transform
            # to map "0D" points back to 1D so that the store looks like 0D
            # to the client.
            sanitized_shape = Shape([1])
            transform = identity.stack(Project(0, 0))
        else:
            sanitized_shape = shape
            transform = None

        storage = Storage(
            sanitized_shape,
            0,
            dtype,
            data=data,
            kind=kind,
        )
        return Store(
            dtype,
            storage,
            transform=transform,
            shape=shape,
            ndim=ndim,
        )

    def create_manual_task(
        self,
        context: Context,
        task_id: int,
        launch_domain: Rect,
    ) -> ManualTask:
        """
        Creates a manual task.

        Parameters
        ----------
        context: Context
            Library to which the task id belongs

        task_id : int
            Task id. Scoped locally within the context; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        launch_domain : Rect, optional
            Launch domain of the task.

        Returns
        -------
        ManualTask
            A new task
        """

        from .operation import ManualTask

        # Check if the task id is valid for this library and the task
        # has the right variant
        machine = context.slice_machine_for_task(task_id)
        unique_op_id = self.get_unique_op_id()
        if launch_domain is None:
            raise RuntimeError(
                "Launch domain must be specified for manual parallelization"
            )

        return ManualTask(
            context,
            task_id,
            launch_domain,
            unique_op_id,
            machine,
        )

    def create_auto_task(
        self,
        context: Context,
        task_id: int,
    ) -> AutoTask:
        """
        Creates an auto task.

        Parameters
        ----------
        context: Context
            Library to which the task id belongs

        task_id : int
            Task id. Scoped locally within the context; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        Returns
        -------
        AutoTask
            A new automatically parallelized task

        See Also
        --------
        Context.create_task
        """

        from .operation import AutoTask

        # Check if the task id is valid for this library and the task
        # has the right variant
        machine = context.slice_machine_for_task(task_id)
        unique_op_id = self.get_unique_op_id()
        return AutoTask(context, task_id, unique_op_id, machine)

    def create_copy(self) -> Copy:
        """
        Creates a copy operation.

        Returns
        -------
        Copy
            A new copy operation
        """

        from .operation import Copy

        return Copy(
            self.core_context,
            self.get_unique_op_id(),
            self.machine,
        )

    def issue_fill(
        self,
        lhs: Store,
        value: Store,
    ) -> None:
        """
        Fills the store with a constant value.

        Parameters
        ----------
        lhs : Store
            Store to fill

        value : Store
            Store holding the constant value to fill the ``lhs`` with

        Raises
        ------
        ValueError
            If the ``value`` is not scalar or the ``lhs`` is either unbound or
            scalar
        """
        from .operation import Fill

        fill = Fill(
            self.core_context,
            lhs,
            value,
            self.get_unique_op_id(),
            self.machine,
        )
        fill.execute()

    def tree_reduce(
        self, context: Context, task_id: int, store: Store, radix: int = 4
    ) -> Store:
        """
        Performs a user-defined reduction by building a tree of reduction
        tasks. At each step, the reducer task gets up to ``radix`` input stores
        and is supposed to produce outputs in a single unbound store.

        Parameters
        ----------
        context : Context
            Library to which the task id belongs

        task_id : int
            Id of the reducer task

        store : Store
            Store to perform reductions on

        radix : int
            Fan-in of each reducer task. If the store is partitioned into
            :math:`N` sub-stores by the runtime, then the first level of
            reduction tree has :math:`\\ceil{N / \\mathtt{radix}}` reducer
            tasks.

        Returns
        -------
        Store
            Store that contains reduction results
        """
        from .operation import Reduce

        if store.ndim > 1:
            raise NotImplementedError(
                "Tree reduction doesn't currently support "
                "multi-dimensional stores"
            )

        result = self.create_store(store.type)

        # A single Reduce operation is mapped to a whole reduction tree
        task = Reduce(
            context,
            task_id,
            radix,
            self.get_unique_op_id(),
            self.machine,
        )
        task.add_input(store)
        task.add_output(result)
        task.execute()
        return result

    def find_region_manager(self, region: Region) -> RegionManager:
        assert region in self.region_managers_by_region
        return self.region_managers_by_region[region]

    def revive_manager(self, region_mgr: RegionManager) -> None:
        self.lru_managers.remove(region_mgr)

    def free_region_manager(
        self, shape: Shape, region: Region, unordered: bool = False
    ) -> None:
        assert region in self.region_managers_by_region
        region_mgr = self.region_managers_by_region[region]
        self.lru_managers.appendleft(region_mgr)

        if len(self.lru_managers) > self._max_lru_length:
            region_mgr = self.lru_managers.pop()
            self.destroy_region_manager(region_mgr, unordered)
        assert len(self.lru_managers) <= self._max_lru_length

    def destroy_region_manager(
        self, region_mgr: RegionManager, unordered: bool
    ) -> None:
        region = region_mgr.region
        del self.region_managers_by_region[region]
        for field_manager in self.field_managers.values():
            field_manager.remove_all_fields(region)

        shape = region_mgr.shape
        active_mgr = self.active_region_managers.get(shape)
        if active_mgr is region_mgr:
            del self.active_region_managers[shape]
        region_mgr.destroy(unordered)

    def find_or_create_region_manager(self, shape: Shape) -> RegionManager:
        region_mgr = self.active_region_managers.get(shape)
        if region_mgr is not None and region_mgr.has_space:
            return region_mgr

        index_space = shape.get_index_space(self)
        field_space = self.create_field_space()
        region = self.create_region(index_space, field_space)

        region_mgr = RegionManager(shape, region)
        self.active_region_managers[shape] = region_mgr
        self.region_managers_by_region[region] = region_mgr
        return region_mgr

    def find_or_create_field_manager(
        self, shape: Shape, field_size: int
    ) -> FieldManager:
        key = (shape, field_size)
        field_mgr = self.field_managers.get(key)
        if field_mgr is not None:
            return field_mgr
        field_mgr = self._field_manager_class(self, shape, field_size)
        self.field_managers[key] = field_mgr
        return field_mgr

    def allocate_field(self, shape: Shape, dtype: Any) -> RegionField:
        from .store import RegionField

        assert not self.destroyed
        region = None
        field_id = None
        field_mgr = self.find_or_create_field_manager(shape, dtype.size)
        region, field_id = field_mgr.allocate_field()
        return RegionField.create(region, field_id, dtype.size, shape)

    def free_field(
        self, region: Region, field_id: int, field_size: int, shape: Shape
    ) -> None:
        # Have a guard here to make sure that we don't try to
        # do this after we have been destroyed
        if self.destroyed:
            return
        # Now save it in our data structure for free fields eligible for reuse
        key = (shape, field_size)
        if key not in self.field_managers:
            return

        self.field_managers[key].free_field(region, field_id)

    def import_output_region(
        self, out_region: OutputRegion, field_id: int, dtype: Any
    ) -> RegionField:
        from .store import RegionField

        region = out_region.get_region()
        shape = Shape(ispace=region.index_space)
        region_mgr = self.region_managers_by_region.get(region)
        if region_mgr is None:
            region_mgr = RegionManager(shape, region, imported=True)
            self.region_managers_by_region[region] = region_mgr
            self.find_or_create_field_manager(shape, dtype.size)

        revived = region_mgr.increase_field_count()
        if revived:
            self.revive_manager(region_mgr)
        return RegionField.create(region, field_id, dtype.size, shape)

    def create_output_region(
        self, fspace: FieldSpace, fields: FieldListLike, ndim: int
    ) -> OutputRegion:
        return OutputRegion(
            self.legion_context,
            self.legion_runtime,
            field_space=fspace,
            fields=fields,
            ndim=ndim,
        )

    def has_attachment(self, alloc: memoryview) -> bool:
        return self._attachment_manager.has_attachment(alloc)

    def find_or_create_index_space(
        self, bounds: Union[tuple[int, ...], Shape, Rect]
    ) -> IndexSpace:
        # Haven't seen this before so make it now
        if isinstance(bounds, Rect):
            rect = bounds
        else:
            rect = Rect(bounds)

        if rect in self.index_spaces:
            return self.index_spaces[rect]
        handle = legion.legion_index_space_create_domain(
            self.legion_runtime, self.legion_context, rect.raw()
        )
        result = IndexSpace(
            self.legion_context, self.legion_runtime, handle=handle
        )
        # Save this for the future
        self.index_spaces[rect] = result
        return result

    def create_field_space(self) -> FieldSpace:
        return FieldSpace(self.legion_context, self.legion_runtime)

    def create_region(
        self, index_space: IndexSpace, field_space: FieldSpace
    ) -> Region:
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

    def extract_scalar(self, future: Future, idx: int) -> Future:
        from .launcher import TaskLauncher

        launcher = TaskLauncher(
            self.core_context,
            self.core_library.LEGATE_CORE_EXTRACT_SCALAR_TASK_ID,
        )
        launcher.add_future(future)
        launcher.add_scalar_arg(idx, ty.int32)
        return launcher.execute_single()

    def extract_scalar_with_domain(
        self, future: FutureMap, idx: int, launch_domain: Rect
    ) -> FutureMap:
        from .launcher import TaskLauncher

        launcher = TaskLauncher(
            self.core_context,
            self.core_library.LEGATE_CORE_EXTRACT_SCALAR_TASK_ID,
        )
        launcher.add_future_map(future)
        launcher.add_scalar_arg(idx, ty.int32)
        return launcher.execute(launch_domain).future_map

    def reduce_future_map(
        self, future_map: Union[Future, FutureMap], redop: int
    ) -> Future:
        if isinstance(future_map, Future):
            return future_map
        else:
            return future_map.reduce(
                self.legion_context,
                self.legion_runtime,
                redop,
                mapper=self.core_context.mapper_id,
            )

    def reduce_exception_future_map(
        self,
        future_map: Union[Future, FutureMap],
    ) -> Future:
        if isinstance(future_map, Future):
            return future_map
        else:
            redop = self.core_library.LEGATE_CORE_JOIN_EXCEPTION_OP
            return future_map.reduce(
                self.legion_context,
                self.legion_runtime,
                self.core_context.get_reduction_op_id(redop),
                mapper=self.core_context.mapper_id,
                tag=self.core_library.LEGATE_CORE_JOIN_EXCEPTION_TAG,
            )

    def issue_execution_fence(self, block: bool = False) -> None:
        """
        Issues an execution fence. A fence is a special operation that
        guarantees that all upstream operations finish before any of the
        downstream operations start. The caller can optionally block on
        completion of all upstream operations.

        Parameters
        ----------
        block : bool
            If ``True``, the call blocks until all upstream operations finish.
        """
        fence = Fence(mapping=False)
        future = fence.launch(self.legion_runtime, self.legion_context)
        if block:
            future.wait()

    def get_nccl_communicator(self) -> Communicator:
        return self._comm_manager.get_nccl_communicator()

    def get_cpu_communicator(self) -> Communicator:
        return self._comm_manager.get_cpu_communicator()

    def delinearize_future_map(
        self, future_map: FutureMap, new_domain: Rect
    ) -> FutureMap:
        ispace = self.find_or_create_index_space(new_domain)
        functor = (
            self.core_library.legate_linearizing_point_transform_functor()
        )
        handle = legion.legion_future_map_transform(
            self.legion_runtime,
            self.legion_context,
            future_map.handle,
            ispace.handle,
            # CFFI constructs a legion_point_transform_functor_t from this list
            [functor],
            False,
        )
        return FutureMap(handle)

    def get_barriers(self, count: int) -> tuple[Future, Future]:
        arrival_barrier = legion.legion_phase_barrier_create(
            self.legion_runtime, self.legion_context, count
        )
        wait_barrier = legion.legion_phase_barrier_advance(
            self.legion_runtime, self.legion_context, arrival_barrier
        )
        # TODO: For now we destroy these barriers during shutdown
        self._barriers.append(arrival_barrier)
        self._barriers.append(wait_barrier)
        return (
            Future.from_cdata(self.legion_runtime, arrival_barrier),
            Future.from_cdata(self.legion_runtime, wait_barrier),
        )

    def record_pending_exception(
        self,
        exn_types: list[type],
        future: Future,
        tb_repr: Optional[str] = None,
    ) -> None:
        exn = PendingException(exn_types, future, tb_repr)
        self._pending_exceptions.append(exn)

    def raise_exceptions(self) -> None:
        pending_exceptions = self._pending_exceptions
        self._pending_exceptions = []
        for pending in pending_exceptions:
            pending.raise_exception()

    def set_provenance(self, provenance: str) -> None:
        """
        Sets a new provenance string

        Parameters
        ----------
        provenance : str
            Provenance string
        """
        self._annotations[-1].set_provenance(provenance)

    def reset_provenance(self) -> None:
        """
        Clears the provenance string that is currently set
        """
        self._annotations[-1].reset_provenance()

    def push_provenance(self, provenance: str) -> None:
        """
        Pushes a provenance string to the stack

        Parameters
        ----------
        provenance : str
            Provenance string
        """
        self._annotations.append(LibraryAnnotations())
        self.set_provenance(provenance)

    def pop_provenance(self) -> None:
        """
        Pops the provenance string on top the stack
        """
        if len(self._annotations) == 1:
            raise ValueError("Provenance stack underflow")
        self._annotations.pop(-1)

    def track_provenance(
        self, func: AnyCallable, nested: bool = False
    ) -> AnyCallable:
        """
        Wraps a function with provenance tracking. Provenance of each operation
        issued within the wrapped function will be tracked automatically.

        Parameters
        ----------
        func : AnyCallable
            Function to wrap

        nested : bool
            If ``True``, each invocation to a wrapped function within another
            wrapped function updates the provenance string. Otherwise, the
            provenance is tracked only for the outermost wrapped function.

        Returns
        -------
        AnyCallable
            Wrapped function
        """
        if nested:

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.push_provenance(caller_frameinfo())
                result = func(*args, **kwargs)
                self.pop_provenance()
                return result

        else:

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if self.provenance is None:
                    self.set_provenance(caller_frameinfo())
                    result = func(*args, **kwargs)
                    self.reset_provenance()
                else:
                    result = func(*args, **kwargs)
                return result

        return wrapper

    def find_or_create_array_type(
        self, elem_type: ty.Dtype, n: int
    ) -> ty.Dtype:
        """
        Finds or creates a fixed array type for a given element type and a
        size.

        Returns the same array type for the same element type and array size.

        Parameters
        ----------
        elem_type : Dtype
            Element type

        n : int
            Array size

        Returns
        -------
        Dtype
            Fixed-size array type unique to each pair of an element type and
            a size
        """
        key = (elem_type, n)
        if key not in self._array_type_cache:
            self._array_type_cache[key] = ty.array_type(elem_type, n)
        return self._array_type_cache[key]


runtime: Runtime = Runtime(core_library)


def _cleanup_legate_runtime() -> None:
    global runtime
    future_leak_check = settings.future_leak_check()
    runtime.destroy()
    del runtime
    gc.collect()
    if future_leak_check:
        print(
            "Looking for cycles that are keeping Future/FutureMap objects "
            "alive after Legate runtime exit."
        )
        find_cycles(True)


add_cleanup_item(_cleanup_legate_runtime)


class _CycleCheckWrapper(ModuleType):
    def __init__(self, wrapped_mod: ModuleType):
        self._wrapped_mod = wrapped_mod

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._wrapped_mod, attr)

    def __del__(self) -> None:
        print(
            "Looking for cycles that are stopping RegionFields from getting "
            "collected and reused."
        )
        find_cycles(False)


if settings.cycle_check():
    # The first thing that legion_top does after executing the user script
    # is to remove the newly created "__main__" module. We intercept this
    # deletion operation to perform our check.
    sys.modules["__main__"] = _CycleCheckWrapper(sys.modules["__main__"])


def get_legion_runtime() -> legion.legion_runtime_t:
    return runtime.legion_runtime


def get_legion_context() -> legion.legion_context_t:
    return runtime.legion_context


def legate_add_library(library: Library) -> None:
    runtime.register_library(library)


def get_legate_runtime() -> Runtime:
    """
    Returns the Legate runtime

    Returns
    -------
        Legate runtime object
    """
    return runtime


def track_provenance(
    nested: bool = False,
) -> Callable[[AnyCallable], AnyCallable]:
    """
    Decorator that adds provenance tracking to functions. Provenance of each
    operation issued within the wrapped function will be tracked automatically.

    Parameters
    ----------
    nested : bool
        If ``True``, each invocation to a wrapped function within another
        wrapped function updates the provenance string. Otherwise, the
        provenance is tracked only for the outermost wrapped function.

    Returns
    -------
    Decorator
        Function that takes a function and returns a one with provenance
        tracking

    See Also
    --------
    legate.core.runtime.Runtime.track_provenance
    """

    def decorator(func: AnyCallable) -> AnyCallable:
        return runtime.track_provenance(func, nested=nested)

    return decorator


class Annotation:
    def __init__(self, pairs: dict[str, str]) -> None:
        """
        Constructs a new annotation object

        Parameters
        ----------
        pairs : dict[str, str]
            Annotations as key-value pairs
        """
        self._annotation = runtime.annotation
        self._pairs = pairs

    def __enter__(self) -> None:
        self._annotation.update(**self._pairs)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for key in self._pairs.keys():
            self._annotation.remove(key)


def get_machine() -> Machine:
    return runtime.machine
