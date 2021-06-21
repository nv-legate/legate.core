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
from functools import partial, reduce

import numpy as np

from .legion import (
    AffineTransform,
    Attach,
    Detach,
    Future,
    InlineMapping,
    Point,
    ffi,
    legion,
)
from .partition import Tiling
from .solver import Shape
from .transform import Project, Promote, Slice


# A region field holds a reference to a field in a logical region
class RegionField(object):
    def __init__(
        self,
        runtime,
        region,
        field,
        shape,
        parent=None,
        transform=None,
        view=None,
    ):
        self.runtime = runtime
        self.attachment_manager = runtime.attachment_manager
        self.partition_manager = runtime.partition_manager
        self.region = region
        self.field = field
        self.shape = shape
        self.parent = parent
        self.transform = transform
        self.key_partition = None  # The key partition for this region field
        self.subviews = None  # RegionField subviews of this region field
        self.view = view  # The view slice tuple used to make this region field
        self.launch_space = None  # Parallel launch space for this region_field
        self.attached_array = (
            None  # Numpy array that we attached to this field
        )
        # Numpy array that we returned for the application
        self.numpy_array = None
        self.physical_region = None  # Physical region for attach
        self.physical_region_refs = 0
        self.physical_region_mapped = False

        self._partitions = {}

    def __del__(self):
        if self.attached_array is not None:
            self.detach_numpy_array(unordered=True, defer=True)

    def has_parallel_launch_space(self):
        return self.launch_space is not None

    def compute_parallel_launch_space(self):
        # See if we computed it already
        if self.launch_space == ():
            return None
        if self.launch_space is not None:
            return self.launch_space
        if self.parent is not None:
            key_partition = self.find_or_create_key_partition()
            if key_partition is None:
                self.launch_space = ()
            else:
                self.launch_space = key_partition.color_shape
        else:  # top-level region so just do the natural partitioning
            self.launch_space = self.partition_manager.compute_parallel_launch_space_by_shape(  # noqa E501
                self.shape
            )
            if self.launch_space is None:
                self.launch_space = ()
        if self.launch_space == ():
            return None
        return self.launch_space

    def set_key_partition(self, part, shardfn=None, shardsp=None):
        assert part.parent == self.region
        self.launch_space = part.color_shape
        self.key_partition = part

    def find_or_create_key_partition(self):
        if self.key_partition is not None:
            return self.key_partition
        # We already tried to compute it and did not have one so we're done
        if self.launch_space == ():
            return None
        if self.parent is not None:
            # Figure out how many tiles we overlap with of the root
            root = self.parent
            while root.parent is not None:
                root = root.parent
            root_key = root.find_or_create_key_partition()
            if root_key is None:
                self.launch_space = ()
                return None
            # Project our bounds through the transform into the
            # root coordinate space to get our bounds in the root
            # coordinate space
            lo = np.zeros((len(self.shape),), dtype=np.int64)
            hi = np.array(self.shape, dtype=np.int64) - 1
            if self.transform:
                lo = self.transform.apply(lo)
                hi = self.transform.apply(hi)
            # Compute the lower bound tile and upper bound tile
            assert len(lo) == len(root_key.tile_shape)
            color_lo = tuple(map(lambda x, y: x // y, lo, root_key.tile_shape))
            color_hi = tuple(map(lambda x, y: x // y, hi, root_key.tile_shape))
            color_tile = root_key.tile_shape
            if self.transform:
                # Check to see if this transform is invertible
                # If it is then we'll reuse the key partition of the
                # root in order to guide how we do the partitioning
                # for this view to maximimize locality. If the transform
                # is not invertible then we'll fall back to doing the
                # standard mapping of the index space
                invertible = True
                for m in range(len(root.shape)):
                    nonzero = False
                    for n in range(len(self.shape)):
                        if self.transform.trans[m, n] != 0:
                            if nonzero:
                                invertible = False
                                break
                            if self.transform.trans[m, n] != 1:
                                invertible = False
                                break
                            nonzero = True
                    if not invertible:
                        break
                if not invertible:
                    # Not invertible so fall back to the standard case
                    launch_space = self.partition_manager.compute_parallel_launch_space_by_shape(  # noqa: E501
                        self.shape
                    )
                    if launch_space == ():
                        return None, None
                    tile_shape = self.partition_manager.compute_tile_shape(
                        self.shape, launch_space
                    )
                    self.key_partition = (
                        self.partition_manager.find_or_create_partition(
                            self.region,
                            launch_space,
                            tile_shape,
                            offset=(0,) * len(launch_space),
                            transform=self.transform,
                        )
                    )
                    return self.key_partition
                # We're invertible so do the standard inversion
                inverse = np.transpose(self.transform.trans)
                # We need to make a make a special sharding functor here that
                # projects the points in our launch space back into the space
                # of the root partitions sharding space
                # First construct the affine mapping for points in our launch
                # space back into the launch space of the root
                # This is the special case where we have a special shard
                # function and sharding space that is different than our normal
                # launch space because it's a subset of the root's launch space
                launch_transform = AffineTransform(
                    len(root.shape), len(self.shape), False
                )
                launch_transform.trans = self.transform.trans
                launch_transform.offset = color_lo
                tile_offset = np.zeros((len(self.shape),), dtype=np.int64)
                for n in range(len(self.shape)):
                    nonzero = False
                    for m in range(len(root.shape)):
                        if inverse[n, m] == 0:
                            continue
                        nonzero = True
                        break
                    if not nonzero:
                        tile_offset[n] = 1
                color_lo = tuple((inverse @ color_lo).flatten())
                color_hi = tuple((inverse @ color_hi).flatten())
                color_tile = tuple(
                    (inverse @ color_tile).flatten() + tile_offset
                )
                # Reset lo and hi back to our space
                lo = np.zeros((len(self.shape),), dtype=np.int64)
                hi = np.array(self.shape, dtype=np.int64) - 1
            # Launch space is how many tiles we have in each dimension
            color_shape = tuple(
                map(lambda x, y: (x - y) + 1, color_hi, color_lo)
            )
            # Check to see if they are all one, if so then we don't even need
            # to bother with making the partition
            volume = reduce(lambda x, y: x * y, color_shape)
            assert volume > 0
            if volume == 1:
                self.launch_space = ()
                # We overlap with exactly one point in the root
                # Therefore just record this point
                self.shard_point = Point(color_lo)
                return None
            # Now compute the offset for the partitioning
            # This will shift the tile down if necessary to align with the
            # boundaries at the root while still covering all of our elements
            offset = tuple(
                map(
                    lambda x, y: 0 if (x % y) == 0 else ((x % y) - y),
                    lo,
                    color_tile,
                )
            )
            self.key_partition = (
                self.partition_manager.find_or_create_partition(
                    self.region,
                    color_shape,
                    color_tile,
                    offset,
                    self.transform,
                )
            )
        else:
            launch_space = self.compute_parallel_launch_space()
            if launch_space is None:
                return None
            tile_shape = self.partition_manager.compute_tile_shape(
                self.shape, launch_space
            )
            self.key_partition = (
                self.partition_manager.find_or_create_partition(
                    self.region,
                    launch_space,
                    tile_shape,
                    offset=(0,) * len(launch_space),
                    transform=self.transform,
                )
            )
        return self.key_partition

    def find_or_create_congruent_partition(
        self, part, transform=None, offset=None
    ):
        if transform is not None:
            shape_transform = AffineTransform(transform.M, transform.N, False)
            shape_transform.trans = transform.trans.copy()
            shape_transform.offset = offset
            offset_transform = transform
            return self.find_or_create_partition(
                shape_transform.apply(part.color_shape),
                shape_transform.apply(part.tile_shape),
                offset_transform.apply(part.tile_offset),
            )
        else:
            assert len(self.shape) == len(part.color_shape)
            return self.find_or_create_partition(
                part.color_shape, part.tile_shape, part.tile_offset
            )

    def find_or_create_partition(
        self, launch_space, tile_shape=None, offset=None
    ):
        # Compute a tile shape based on our shape
        if tile_shape is None:
            tile_shape = self.partition_manager.compute_tile_shape(
                self.shape, launch_space
            )
        if offset is None:
            offset = (0,) * len(launch_space)
        # Tile shape should have the same number of dimensions as our shape
        assert len(launch_space) == len(self.shape)
        assert len(tile_shape) == len(self.shape)
        assert len(offset) == len(self.shape)
        # Do a quick check to see if this is congruent to our key partition
        if (
            self.key_partition is not None
            and launch_space == self.key_partition.color_shape
            and tile_shape == self.key_partition.tile_shape
            and offset == self.key_partition.tile_offset
        ):
            return self.key_partition
        # Continue this process on the region object, to ensure any created
        # partitions are shared between RegionField objects referring to the
        # same region
        return self.partition_manager.find_or_create_partition(
            self.region,
            launch_space,
            tile_shape,
            offset,
            self.transform,
        )

    def find_or_create_indirect_partition(self, launch_space):
        assert len(launch_space) != len(self.shape)
        # If there is a mismatch in the number of dimensions then we need
        # to compute a partition and projection functor that can transform
        # the points into a partition that makes sense
        raise NotImplementedError("need support for indirect partitioning")

    def attach_numpy_array(self, context, numpy_array, share=False):
        assert self.parent is None
        assert isinstance(numpy_array, np.ndarray)
        # If we already have a numpy array attached
        # then we have to detach it first
        if self.attached_array is not None:
            if self.attached_array is numpy_array:
                return
            else:
                self.detach_numpy_array(unordered=False)
        # Now we can attach the new one and then do the acquire
        attach = Attach(
            self.region,
            self.field.field_id,
            numpy_array,
            mapper=context.mapper_id,
        )
        # If we're not sharing then there is no need to map or restrict the
        # attachment
        if not share:
            # No need for restriction for us
            attach.set_restricted(False)
            # No need for mapping in the restricted case
            attach.set_mapped(False)
        else:
            self.physical_region_mapped = True
        self.physical_region = self.runtime.dispatch(attach)
        # Due to the working of the Python interpreter's garbage collection
        # algorithm we make the detach operation for this now and register it
        # with the runtime so that we know that it won't be collected when the
        # RegionField object is collected
        detach = Detach(self.physical_region, flush=True)
        # Dangle these fields off here to prevent premature collection
        detach.field = self.field
        detach.array = numpy_array
        self.detach_key = self.attachment_manager.register_detachment(detach)
        # Add a reference here to prevent collection in for inline mapped cases
        assert self.physical_region_refs == 0
        # This reference will never be removed, we'll delete the
        # physical region once the object is deleted
        self.physical_region_refs = 1
        self.attached_array = numpy_array
        if share:
            # If we're sharing this then we can also make this our numpy array
            self.numpy_array = weakref.ref(numpy_array)

    def detach_numpy_array(self, unordered, defer=False):
        assert self.parent is None
        assert self.attached_array is not None
        assert self.physical_region is not None
        detach = self.attachment_manager.remove_detachment(self.detach_key)
        detach.unordered = unordered
        self.attachment_manager.detach_array(
            self.attached_array, self.field, detach, defer
        )
        self.physical_region = None
        self.physical_region_mapped = False
        self.attached_array = None

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

    def decrement_inline_mapped_ref_count(self):
        if self.parent is None:
            assert self.physical_region_refs > 0
            self.physical_region_refs -= 1
            if self.physical_region_refs == 0:
                self.runtime.unmap_region(self.physical_region)
                self.physical_region = None
                self.physical_region_mapped = False
        else:
            self.parent.decrement_inline_mapped_ref_count()

    def get_numpy_array(self, shape, context=None, transform=None):
        context = self.runtime.context if context is None else context

        # See if we still have a valid numpy array to use
        if self.numpy_array is not None:
            # Test the weak reference to see if it is still alive
            result = self.numpy_array()
            if result is not None:
                return result
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
        initializer = _RegionNdarray(
            tuple(shape), self.field.dtype, base_ptr, strides, False
        )
        array = np.asarray(initializer)

        # This will be the unmap call that will be invoked once the weakref is
        # removed
        # We will use it to unmap the inline mapping that was performed
        def decrement(region_field, ref):
            region_field.decrement_inline_mapped_ref_count()

        # Curry bind arguments to the function
        callback = partial(decrement, self)
        # Save a weak reference to the array so we don't prevent collection
        self.numpy_array = weakref.ref(array, callback)
        return array

    def get_tile(self, shape, tiling):
        tile_shape = tiling.tile_shape
        # As an interesting optimization, if we can evenly tile the
        # region in all dimensions of the parent region, then we'll make
        # a disjoint tiled partition with as many children as possible
        complete_tiling = (tiling.offset % tile_shape).volume == 0

        if complete_tiling and self.partition_manager.use_complete_tiling(
            self, tile_shape
        ):
            color_shape = shape // tile_shape
            tiling = Tiling(self.runtime, tile_shape, color_shape)
            color = tiling.offset // tile_shape
        else:
            color = (0,) * shape.ndim

        if tiling in self._partitions:
            partition = self._partitions[tiling]
        else:
            partition = tiling.construct(self.region, shape, complete_tiling)
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
class _RegionNdarray(object):
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
        shape,
        dtype,
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
        dtype : numpy.dtype
            Data type of this store
        storage : Any[RegionField, Future, FutureMap]
            A backing storage of this store
        optimize_scalar : bool
            Whether to use a Future for the storage when the volume is 1

        """
        self._runtime = runtime
        self._shape = shape
        self._dtype = dtype
        assert (
            storage is None
            or isinstance(storage, RegionField)
            or isinstance(storage, Future)
        )
        self._storage = storage
        self._scalar = optimize_scalar and shape.volume <= 1
        self._parent = parent
        self._transform = transform
        self._accessor_transform = None

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._shape.ndim

    @property
    def type(self):
        """
        Return the type of the data in this storage primitive
        """
        return self._dtype

    @property
    def kind(self):
        """
        Return the type of the Legion storage object backing the
        data in this storage object: either Future, FutureMap,
        RegionField
        """
        if self._storage is not None:
            return type(self._storage)
        else:
            return Future if self._scalar else RegionField

    @property
    def storage(self):
        """
        Return the Legion storage objects actually backing the
        data for this Store. These will have exactly the
        type specified in by 'kind'
        """
        if self._storage is None:
            # TODO: We should keep track of thunks and evaluate them
            #       if necessary
            if self._parent is None:
                if self._scalar:
                    raise ValueError(
                        "Illegal to access the storage of an uninitialized "
                        "Legate store of volume 1 with scalar optimization"
                    )
                else:
                    self._storage = self._runtime.allocate_field(
                        self._shape, self._dtype
                    )
            else:
                assert self._transform is not None
                if self._parent.kind == Future:
                    self._storage = self._parent._storage
                    return

                tiling = Tiling(
                    self._runtime, self.shape, Shape((1,) * self.ndim)
                )
                self._storage = self._get_tile(tiling)

        return self._storage

    def _get_tile(self, tiling):
        if self._parent is not None:
            tiling = self._transform.invert(tiling)
            return self._parent._get_tile(tiling)
        else:
            # If the tile covers the entire region, we don't need to create
            # a subregion
            if self.shape == tiling.tile_shape:
                return self.storage
            else:
                return self.storage.get_tile(self.shape, tiling)

    def __str__(self):
        if self._parent is None:
            return (
                f"<Store(shape: {self._shape}, type: {self._dtype}, "
                f"kind: {self.kind.__name__}, storage: {self._storage.shape})>"
            )
        else:
            return (
                f"<Store(shape: {self._shape}, type: {self._dtype}, "
                f"kind: {self.kind.__name__})> <<=={self._transform}== "
                f"{self._parent}"
            )

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
            shape,
            self._dtype,
            optimize_scalar=self._scalar,
            parent=self,
            transform=transform,
        )

    # Take a hyperplane of an N-D store for a given index
    # to create an (N-1)-D store
    def project(self, dim, index):
        transform = Project(self._runtime, dim, index)
        shape = transform.compute_shape(self._shape)
        if self._shape == shape:
            return self
        return Store(
            self._runtime,
            shape,
            self._dtype,
            optimize_scalar=self._scalar,
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

        transform = Slice(self._runtime, dim, slice(start, stop, step))
        shape = transform.compute_shape(self._shape)
        if self._shape == shape:
            return self
        return Store(
            self._runtime,
            shape,
            self._dtype,
            optimize_scalar=self._scalar,
            parent=self,
            transform=transform,
        )

    def get_accessor_transform(self):
        if self._parent is None:
            return None
        else:
            if self._accessor_transform is None:
                self._accessor_transform = (
                    self._transform.get_accessor_transform(
                        self.shape,
                        self._parent.get_accessor_transform(),
                    )
                )
            return self._accessor_transform

    def get_numpy_array(self, context=None):
        transform = self.get_accessor_transform()
        return self.storage.get_numpy_array(
            self.shape, context=context, transform=transform
        )
