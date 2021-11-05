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

import weakref

from .legion import (
    Attach,
    Detach,
    Future,
    IndexAttach,
    IndexDetach,
    InlineMapping,
    Point,
    ffi,
    legion,
)
from .partition import NoPartition, Restriction, Tiling
from .shape import Shape
from .transform import (
    Delinearize,
    NonInvertibleError,
    Project,
    Promote,
    Shift,
    Transpose,
)
from .types import _Dtype


class InlineMappedAllocation(object):
    """
    This helper class is to tie the lifecycle of the client object to
    the inline mapped allocation
    """

    def __init__(self, region_field, shape, address, strides):
        self._region_field = region_field
        self._shape = shape
        self._address = address
        self._strides = strides
        self._consumed = False

    def consume(self, ctor):
        if self._consumed:
            raise RuntimeError("Each inline mapping can be consumed only once")
        self._consumed = True
        result = ctor(self._shape, self._address, self._strides)
        self._region_field.register_consumer(result)
        return result


class DistributedAllocation(object):
    def __init__(self, partition, shard_local_buffers):
        """
        Represents a distributed collection of buffers, to be
        collectively attached as sub-regions of the same
        parent region.

        This is a rare case of a data structure that is allowed (and expected)
        to have a different value on different shards; each shard should
        specify a distinct set of resources.

        Parameters
        ----------
        partition : Partition
            The partition to use in the IndexAttach operation
        shard_local_buffers : dict[Point, memoryview]
            Map from color to buffer that should back the sub-region of that
            color. This map will only cover the buffers local to the current
            shard.
        """
        self.partition = partition
        self.shard_local_buffers = shard_local_buffers


# A region field holds a reference to a field in a logical region
class RegionField(object):
    def __init__(
        self,
        runtime,
        region,
        field,
        shape,
        parent=None,
    ):
        self.runtime = runtime
        self.attachment_manager = runtime.attachment_manager
        self.partition_manager = runtime.partition_manager
        self.region = region
        self.field = field
        self.shape = shape
        self.parent = parent
        self.launch_space = None  # Parallel launch space for this region_field
        # External allocation we attached to this field
        self.attached_alloc = None
        self.detach_key = None
        self.physical_region = None  # Physical region for attach
        self.physical_region_refs = 0
        self.physical_region_mapped = False

        self._partitions = {}

    def __del__(self):
        if self.attached_alloc is not None:
            self.detach_external_allocation(unordered=True, defer=True)

    def same_handle(self, other):
        return (
            type(self) == type(other)
            and self.region.same_handle(other.region)
            and self.field.same_handle(other.field)
        )

    def __str__(self):
        return (
            f"RegionField("
            f"tid: {self.region.handle.tree_id}, "
            f"is: {self.region.handle.index_space.id}, "
            f"fs: {self.region.handle.field_space.id}, "
            f"fid: {self.field.field_id})"
        )

    def compute_parallel_launch_space(self):
        # See if we computed it already
        if self.launch_space == ():
            return None
        if self.launch_space is not None:
            return self.launch_space
        self.launch_space = self.partition_manager.compute_parallel_launch_space_by_shape(  # noqa E501
            self.shape
        )
        if self.launch_space is None:
            self.launch_space = ()
        if self.launch_space == ():
            return None
        return self.launch_space

    def attach_external_allocation(self, context, alloc, share):
        assert self.parent is None
        # If we already have some memory attached, detach it first
        if self.attached_alloc is not None:
            return RuntimeError("A RegionField cannot be re-attached")
        # All inline mappings should have been unmapped by now
        assert self.physical_region_refs == 0
        # Record the attached memory ranges, and confirm no overlaps with
        # previously encountered ranges.
        self.attachment_manager.attach_external_allocation(alloc, self)
        if isinstance(alloc, memoryview):
            # Singleton attachment
            attach = Attach(
                self.region,
                self.field.field_id,
                alloc,
                mapper=context.mapper_id,
            )
            # If we're not sharing then there is no need to map or restrict the
            # attachment
            if not share:
                attach.set_restricted(False)
                attach.set_mapped(False)
            else:
                self.physical_region_mapped = True
            # Singleton allocations return a physical region for the entire
            # domain, that can be inline-mapped directly.
            self.physical_region = self.runtime.dispatch(attach)
            # Add a reference here to prevent collection in inline mapped
            # cases. This reference will never be removed, we'll delete the
            # physical region once the object is deleted.
            self.physical_region_refs = 1
            # Due to the working of the Python interpreter's garbage collection
            # algorithm we make the detach operation for this now and register
            # it with the runtime so that we know that it won't be collected
            # when the RegionField object is collected.
            # We don't need to flush the contents back to the attached memory
            # if this is an internal temporary allocation.
            detach = Detach(self.physical_region, flush=share)
        else:
            # Distributed attachment
            assert alloc.partition.parent is self.region
            field_type = self.region.field_space.get_type(self.field.field_id)
            field_size = (
                field_type.size
                if isinstance(field_type, _Dtype)
                else field_type
            )
            shard_local_data = {}
            for (c, buf) in alloc.shard_local_buffers.items():
                subregion = alloc.partition.get_child(c)
                bounds = subregion.index_space.get_bounds()
                if buf.shape != tuple(
                    bounds.hi[i] - bounds.lo[i] + 1 for i in range(bounds.dim)
                ):
                    raise RuntimeError(
                        "Subregion shape does not match attached buffer"
                    )
                if buf.itemsize != field_size:
                    raise RuntimeError(
                        "Field type does not match attached buffer"
                    )
                shard_local_data[subregion] = buf
            attach = IndexAttach(
                self.region,
                self.field.field_id,
                shard_local_data,
                mapper=context.mapper_id,
            )
            attach.set_deduplicate_across_shards(True)
            # If we're not sharing there is no need to restrict the attachment
            if not share:
                attach.set_restricted(False)
            external_resources = self.runtime.dispatch(attach)
            # We don't need to flush the contents back to the attached memory
            # if this is an internal temporary allocation.
            detach = IndexDetach(external_resources, flush=share)
        # Record the attachment
        self.attached_alloc = alloc
        # Dangle these fields off the detachment operation, to prevent
        # premature collection
        detach.field = self.field
        detach.alloc = alloc
        # Don't store the detachment operation here, instead register it on the
        # attachment manager and record its unique key
        # TODO: This might not be necessary anymore
        self.detach_key = self.attachment_manager.register_detachment(detach)

    def detach_external_allocation(self, unordered, defer=False):
        assert self.parent is None
        assert self.attached_alloc is not None
        detach = self.attachment_manager.remove_detachment(self.detach_key)
        detach.unordered = unordered
        self.attachment_manager.detach_external_allocation(
            self.attached_alloc, detach, defer
        )
        self.physical_region = None
        self.physical_region_mapped = False
        self.physical_region_refs = 0
        self.attached_alloc = None

    def get_inline_mapped_region(self, context):
        if self.parent is None:
            if self.physical_region is None:
                # We don't have a valid numpy array so we need to do an inline
                # mapping and then use the buffer to share the storage
                mapping = InlineMapping(
                    self.region,
                    self.field.field_id,
                    mapper=context.mapper_id,
                )
                self.physical_region = self.runtime.dispatch(mapping)
                self.physical_region_mapped = True
                # Wait until it is valid before returning
                self.physical_region.wait_until_valid()
            elif not self.physical_region_mapped:
                # If we have a physical region but it is not mapped then
                # we actually need to remap it, we do this by launching it
                self.runtime.dispatch(self.physical_region)
                self.physical_region_mapped = True
                # Wait until it is valid before returning
                self.physical_region.wait_until_valid()
            # Increment our ref count so we know when it can be collected
            self.physical_region_refs += 1
            return self.physical_region
        else:
            return self.parent.get_inline_mapped_region(context)

    def decrement_inline_mapped_ref_count(self, unordered=False):
        if self.parent is None:
            if self.physical_region is None:
                return
            assert self.physical_region_refs > 0
            self.physical_region_refs -= 1
            if self.physical_region_refs == 0:
                self.runtime.unmap_region(
                    self.physical_region, unordered=unordered
                )
                self.physical_region = None
                self.physical_region_mapped = False
        else:
            self.parent.decrement_inline_mapped_ref_count(unordered=unordered)

    def get_inline_allocation(self, shape, context=None, transform=None):
        context = self.runtime.context if context is None else context

        physical_region = self.get_inline_mapped_region(context)
        # We need a pointer to the physical allocation for this physical region
        dim = len(shape)
        # Build the accessor for this physical region
        if transform is not None:
            # We have a transform so build the accessor special with a
            # transform
            func = getattr(
                legion,
                "legion_physical_region_get_field_accessor_array_"
                f"{dim}d_with_transform",
            )
            accessor = func(
                physical_region.handle,
                ffi.cast("legion_field_id_t", self.field.field_id),
                transform.raw(),
            )
        else:
            # No transfrom so we can do the normal thing
            func = getattr(
                legion,
                f"legion_physical_region_get_field_accessor_array_{dim}d",
            )
            accessor = func(
                physical_region.handle,
                ffi.cast("legion_field_id_t", self.field.field_id),
            )
        # Now that we've got our accessor we can get a pointer to the memory
        rect = ffi.new(f"legion_rect_{dim}d_t *")
        for d in range(dim):
            rect[0].lo.x[d] = 0
            rect[0].hi.x[d] = shape[d] - 1  # inclusive
        subrect = ffi.new(f"legion_rect_{dim}d_t *")
        offsets = ffi.new("legion_byte_offset_t[]", dim)
        func = getattr(legion, f"legion_accessor_array_{dim}d_raw_rect_ptr")
        base_ptr = func(accessor, rect[0], subrect, offsets)
        assert base_ptr is not None
        # Check that the subrect is the same as in the in rect
        for d in range(dim):
            assert rect[0].lo.x[d] == subrect[0].lo.x[d]
            assert rect[0].hi.x[d] == subrect[0].hi.x[d]
        strides = tuple(offsets[i].offset for i in range(dim))
        # Numpy doesn't know about CFFI pointers, so we have to cast
        # this to a Python long before we can hand it off to Numpy.
        base_ptr = int(ffi.cast("size_t", base_ptr))
        return InlineMappedAllocation(
            self,
            tuple(shape),
            base_ptr,
            strides,
        )

    def register_consumer(self, consumer):
        # We add a callback that will be triggered when the consumer object is
        # collected. This callback carries a (captured) reference to the source
        # RegionField, keeping it alive while any consumers remain. Note that
        # weakref.ref() would not work for this purpose, because callbacks
        # passed to weakref.ref() do NOT keep their pointed objects alive. We
        # avoid storing references from the source RegionField to the consumer,
        # so that we don't create reference cycles.

        def callback():
            self.decrement_inline_mapped_ref_count()

        weakref.finalize(consumer, callback)

    def get_tile(self, shape, tiling):
        tile_shape = tiling.tile_shape
        # As an interesting optimization, if we can evenly tile the
        # region in all dimensions of the parent region, then we'll make
        # a disjoint tiled partition with as many children as possible
        can_tile_completely = (tiling.offset % tile_shape).sum() == 0 and (
            shape % tile_shape
        ).sum() == 0

        if can_tile_completely and self.partition_manager.use_complete_tiling(
            shape, tile_shape
        ):
            color_shape = shape // tile_shape
            new_tiling = Tiling(self.runtime, tile_shape, color_shape)
            color = tiling.offset // tile_shape
            tiling = new_tiling
            complete = True
        else:
            color = (0,) * shape.ndim
            complete = False

        if tiling in self._partitions:
            partition = self._partitions[tiling]
        else:
            partition = tiling.construct(self.region, complete=complete)
            self._partitions[tiling] = partition

        child_region = partition.get_child(Point(color))
        return RegionField(
            self.runtime,
            child_region,
            self.field,
            tiling.tile_shape,
            self,
        )


# This is a dummy object that is only used as an initializer for the
# RegionField object above. It is thrown away as soon as the
# RegionField is constructed.
class _LegateNDarray(object):
    __slots__ = ["__array_interface__"]

    def __init__(self, shape, field_type, base_ptr, strides, read_only):
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        self.__array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": field_type.str,
            "data": (base_ptr, read_only),
            "strides": strides,
        }


class Store(object):
    def __init__(
        self,
        runtime,
        dtype,
        shape=None,
        storage=None,
        optimize_scalar=False,
        parent=None,
        transform=None,
    ):
        """
        Unlike in Arrow where all data is backed by objects that
        implement the Python Buffer protocol, in Legate data is backed
        by Legate Store objects. A Store is a logical unit of storage
        backed by one of three containers: RegionField, Future, or FutureMap.
        Backing storages are materialized lazily and may not be created, if
        the lazy evaluation logic optimized them away.

        Parameters
        ----------
        runtime : legate.core.Runtime
            The Legate runtime
        shape : legate.core.Shape
            A Shape object representing the shape of this store
        dtype : legate.core.types._DType
            Data type of this store
        storage : Any[RegionField, Future, FutureMap]
            A backing storage of this store
        optimize_scalar : bool
            Whether to use a Future for the storage when the volume is 1

        """
        assert isinstance(shape, Shape) or shape is None
        self._runtime = runtime
        self._attachment_manager = runtime.attachment_manager
        self._partition_manager = runtime.partition_manager
        self._shape = shape
        self._dtype = dtype
        assert (
            storage is None
            or isinstance(storage, RegionField)
            or isinstance(storage, Future)
        )
        self._storage = storage
        assert (parent is None and transform is None) or (
            parent is not None and transform is not None
        )
        self._parent = parent
        self._transform = transform
        if isinstance(storage, Future):
            assert shape is not None
            assert self.get_root().shape.volume() <= 1
            self._kind = Future
        elif isinstance(storage, RegionField):
            self._kind = RegionField
        elif parent is not None:
            self._kind = parent._kind
        elif optimize_scalar and shape is not None and shape.volume() <= 1:
            self._kind = Future
        else:
            self._kind = RegionField
        self._inverse_transform = None
        self._partitions = {}
        self._key_partition = None

        if not self.unbound:
            if any(extent < 0 for extent in self._shape.extents):
                raise ValueError(f"Invalid shape: {self._shape}")

    @property
    def shape(self):
        if self._shape is None:
            # If someone wants to access the shape of an unbound
            # store before it is set, that means the producer task is
            # sitting in the queue, so we should flush the queue.
            self._runtime.flush_scheduling_window()
            # At this point, we should have the shape set.
            assert self._shape is not None
        return self._shape

    @property
    def ndim(self):
        return -1 if self._shape is None else self._shape.ndim

    @property
    def type(self):
        """
        Return the type of the data in this storage primitive
        """
        return self._dtype

    def get_dtype(self):
        return self._dtype

    @property
    def kind(self):
        """
        Return the type of the Legion storage object backing the data in this
        storage object: either Future, FutureMap, or RegionField.
        """
        return self._kind

    @property
    def unbound(self):
        return self._shape is None

    @property
    def scalar(self):
        return self._kind is Future and self._shape.volume() == 1

    @property
    def storage(self):
        """
        Return the Legion storage objects actually backing the data for this
        Store. These will have exactly the type specified by `.kind`.
        """
        # If someone is trying to retreive the storage of a store,
        # we need to execute outstanding operations so that we know
        # it has been initialized correctly.
        self._runtime.flush_scheduling_window()
        if self._storage is None:
            if self.unbound:
                raise RuntimeError(
                    "Storage of a variable size store cannot be retrieved "
                    "until it is passed to an operation"
                )
            # TODO: We should keep track of thunks and evaluate them
            #       if necessary
            if self._parent is None:
                if self._kind is Future:
                    raise ValueError(
                        "Illegal to access the storage of an uninitialized "
                        "Legate store of volume 1 with scalar optimization"
                    )
                else:
                    self._storage = self._runtime.allocate_field(
                        self._shape, self._dtype
                    )
            elif self._kind is Future:
                self._storage = self._parent.storage
            else:
                tiling = Tiling(
                    self._runtime, self.shape, Shape((1,) * self.ndim)
                )
                self._storage = self._get_tile(tiling)

        return self._storage

    @property
    def has_storage(self):
        return self._storage is not None or (
            self._parent is not None and self._parent.has_storage
        )

    def attach_external_allocation(self, context, alloc, share):
        if not isinstance(alloc, memoryview) and not isinstance(
            alloc, DistributedAllocation
        ):
            raise ValueError(
                f"Only a memoryview or DistributedAllocation object can be "
                f"attached, but got {alloc}"
            )
        if self._parent is not None:
            raise ValueError("Can only attach buffers to top-level Stores")
        if self._kind is not RegionField:
            raise ValueError(
                "Can only attach buffers to RegionField-backed Stores"
            )
        if self.unbound:
            raise ValueError("Cannot attach buffers to variable-size stores")
        # If the storage has not been set, and this is a non-temporary
        # singleton attachment, we can reuse an existing RegionField that was
        # previously attached to this buffer.
        # This is the only situation where we can attach the same buffer to
        # two Stores, since they are both backed by the same RegionField.
        if self._storage is None and share and isinstance(alloc, memoryview):
            self._storage = self._attachment_manager.reuse_existing_attachment(
                alloc
            )
            if self._storage is not None:
                return
        # Force the RegionField to be instantiated, do the attachment normally
        self.storage.attach_external_allocation(context, alloc, share)

    def get_root(self):
        if self._parent is None:
            return self
        else:
            return self._parent.get_root()

    def comm_volume(self):
        my_tile = self._get_tile_shape()
        return my_tile.tile_shape.volume()

    def set_storage(self, storage):
        assert type(storage) is self._kind
        self._storage = storage
        if self._shape is None:
            assert isinstance(storage, RegionField)
            self._shape = storage.shape
        else:
            assert isinstance(storage, Future)

    def invert_partition(self, partition):
        if self._parent is not None:
            partition = self._transform.invert(partition)
            return self._parent.invert_partition(partition)
        else:
            return partition

    def _get_tile_shape(self):
        tile = Tiling(self._runtime, self.shape, Shape((1,) * self.ndim))
        return self.invert_partition(tile)

    def _get_tile(self, tiling):
        if self._parent is not None:
            try:
                tiling = self._transform.invert(tiling)
            except NonInvertibleError:
                raise RuntimeError(
                    "This slice corresponds to a non-contiguous subset of the "
                    "original store before transformation. Please make a copy "
                    "of the transformed store and slice that copy instead."
                )

            return self._parent._get_tile(tiling)
        else:
            # If the tile covers the entire region, we don't need to create
            # a subregion
            if self.shape == tiling.tile_shape:
                return self.storage
            else:
                return self.storage.get_tile(self.shape, tiling)

    def __str__(self):
        storage = (
            f"{self._kind.__name__}(uninitialized)"
            if self._storage is None
            else str(self._storage)
        )
        result = (
            f"Store("
            f"shape: {self._shape}, "
            f"type: {self._dtype}, "
            f"storage: {storage})"
        )
        if self._parent is not None:
            result += f" <<=={self._transform}== {self._parent}"
        return result

    def __repr__(self):
        return str(self)

    # Convert a store in N-D space to that in (N+1)-D space.
    # The extra_dim specifies the added dimension
    def promote(self, extra_dim, dim_size=1):
        transform = Promote(self._runtime, extra_dim, dim_size)
        shape = transform.compute_shape(self._shape)
        if self._shape == shape:
            return self
        return Store(
            self._runtime,
            self._dtype,
            shape=shape,
            storage=None,
            parent=self,
            transform=transform,
        )

    # Take a hyperplane of an N-D store for a given index
    # to create an (N-1)-D store
    def project(self, dim, index):
        assert dim < self.ndim
        transform = Project(self._runtime, dim, index)
        shape = transform.compute_shape(self._shape)
        if self._shape == shape:
            return self
        return Store(
            self._runtime,
            self._dtype,
            shape=shape,
            storage=None,
            parent=self,
            transform=transform,
        )

    def slice(self, dim, sl):
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Invalid dimension {dim} for a {self.ndim}-D store"
            )

        size = self.shape[dim]
        start = 0 if sl.start is None else sl.start
        stop = size if sl.stop is None else sl.stop
        start = start + size if start < 0 else start
        stop = stop + size if stop < 0 else stop
        step = 1 if sl.step is None else sl.step

        if step != 1:
            raise ValueError(f"Unsupported slicing: {sl}")
        if start >= size or stop > size:
            raise ValueError(f"Out of bounds: {sl} for a shape {self.shape}")

        transform = Shift(self._runtime, dim, -start)
        shape = self._shape.update(dim, stop - start)
        if self._shape == shape:
            return self
        return Store(
            self._runtime,
            self._dtype,
            shape=shape,
            storage=None,
            parent=self,
            transform=transform,
        )

    def transpose(self, axes):
        if len(axes) != self.ndim:
            raise ValueError(
                f"dimension mismatch: expected {self.ndim} axes, "
                f"but got {len(axes)}"
            )
        elif len(axes) != len(set(axes)):
            raise ValueError(f"duplicate axes found: {axes}")

        if all(idx == val for idx, val in enumerate(axes)):
            return self

        transform = Transpose(self._runtime, axes)
        shape = transform.compute_shape(self._shape)
        return Store(
            self._runtime,
            self._dtype,
            shape=shape,
            storage=self._storage if self._kind is Future else None,
            parent=self,
            transform=transform,
        )

    def delinearize(self, dim, shape):
        if len(shape) == 1:
            return self
        s = Shape(shape)
        transform = Delinearize(self._runtime, dim, s)
        if self._shape[dim] != s.volume():
            raise ValueError(
                f"Dimension of size {self._shape[dim]} "
                f"cannot be delinearized into {shape}"
            )
        shape = transform.compute_shape(self._shape)
        return Store(
            self._runtime,
            self._dtype,
            shape=shape,
            storage=None,
            parent=self,
            transform=transform,
        )

    def get_inverse_transform(self):
        if self._parent is None:
            return None
        else:
            if self._inverse_transform is None:
                self._inverse_transform = (
                    self._transform.get_inverse_transform(
                        self.shape,
                        self._parent.get_inverse_transform(),
                    )
                )
            return self._inverse_transform

    def get_inline_allocation(self, context=None):
        assert self._kind is RegionField
        transform = self.get_inverse_transform()
        return self.storage.get_inline_allocation(
            self.shape, context=context, transform=transform
        )

    def overlaps(self, other):
        my_root = self.get_root()
        other_root = other.get_root()
        if my_root is not other_root:
            return False

        my_tile = self._get_tile_shape()
        other_tile = other._get_tile_shape()

        return my_tile.overlaps(other_tile)

    def _serialize_transform(self, buf):
        if self._parent is not None:
            self._transform.serialize(buf)
            self._parent._serialize_transform(buf)
        else:
            buf.pack_32bit_int(-1)

    def serialize(self, buf):
        buf.pack_bool(self._kind is Future)
        buf.pack_32bit_int(self.ndim)
        buf.pack_32bit_int(self._dtype.code)
        self._serialize_transform(buf)

    def has_key_partition(self, restrictions):
        if (
            self._key_partition is not None
            and self._key_partition.satisfies_restriction(restrictions)
        ):
            return True
        elif self._parent is not None and self._transform.invertible:
            restrictions = self._transform.invert_restrictions(restrictions)
            return self._parent.has_key_partition(restrictions)
        else:
            return False

    def set_key_partition(self, key_partition):
        self._key_partition = key_partition

    def reset_key_partition(self):
        self._key_partition = None

    def compute_key_partition(self, restrictions):
        # If this is effectively a scalar store, we don't need to partition it
        if self._kind is Future or self.ndim == 0:
            return NoPartition()

        if (
            self._key_partition is not None
            and self._key_partition.satisfies_restriction(restrictions)
        ):
            return self._key_partition
        elif self._parent is not None and self._transform.invertible:
            restrictions = self._transform.invert_restrictions(restrictions)
            partition = self._parent.compute_key_partition(restrictions)
            return self._transform.convert(partition)

        launch_shape = self._partition_manager.compute_launch_shape(
            self,
            restrictions,
        )
        if launch_shape is None:
            return NoPartition()
        else:
            tile_shape = self._partition_manager.compute_tile_shape(
                self.shape, launch_shape
            )
            return Tiling(self._runtime, tile_shape, launch_shape)

    def _invert_dimensions(self, dims):
        if self._parent is None:
            return dims
        else:
            dims = self._transform.invert_dimensions(dims)
            return self._parent._invert_dimensions(dims)

    def _compute_projection(self, partition):
        dims = self._invert_dimensions(tuple(range(self.ndim)))
        if len(dims) == self.ndim and all(
            idx == dim for idx, dim in enumerate(dims)
        ):
            return 0
        else:
            return self._runtime.get_projection(self.ndim, dims)

    def find_restrictions(self):
        if self._parent is None:
            return (Restriction.UNRESTRICTED,) * self.ndim
        else:
            restrictions = self._parent.find_restrictions()
            return self._transform.convert_restrictions(restrictions)

    def find_or_create_partition(self, functor):
        assert self._kind is RegionField
        if functor in self._partitions:
            return self._partitions[functor]

        # Convert the partition to use the root's coordinate space
        converted = self.invert_partition(functor)
        complete = converted.is_complete_for(self._get_tile_shape())

        # Then, find the right projection functor that maps points in the color
        # space of the child's partition to subregions of the converted
        # partition
        proj = self._compute_projection(converted)

        part = converted.construct(self.storage.region, complete=complete)
        self._partitions[functor] = (part, proj)
        return part, proj
