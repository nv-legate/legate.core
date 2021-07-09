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
from functools import partial

import numpy as np

import legate.core.types as ty

from .legion import Attach, Detach, Future, InlineMapping, Point, ffi, legion
from .partition import NoPartition, Tiling
from .shape import Shape
from .transform import Delinearize, Project, Promote, Shift, Transpose


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
            tuple(shape), self.field.dtype.type, base_ptr, strides, False
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
            partition = tiling.construct(self.region, shape, complete=complete)
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
        unbound=False,
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
        self._partition_manager = runtime.partition_manager
        shape = None if shape is None else Shape(shape)
        self._shape = shape
        self._dtype = dtype
        assert (
            storage is None
            or isinstance(storage, RegionField)
            or isinstance(storage, Future)
        )
        self._storage = storage
        self._unbound = unbound
        self._scalar = (
            optimize_scalar
            and shape is not None
            and (shape.volume() <= 1 or isinstance(storage, Future))
        )
        self._parent = parent
        self._transform = transform
        self._inverse_transform = None
        self._partitions = {}

    @property
    def shape(self):
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
        Return the type of the Legion storage object backing the
        data in this storage object: either Future, FutureMap,
        RegionField
        """
        if self._storage is not None:
            return type(self._storage)
        else:
            return Future if self._scalar else RegionField

    @property
    def unbound(self):
        return self._unbound

    @property
    def scalar(self):
        return self._scalar

    @property
    def storage(self):
        """
        Return the Legion storage objects actually backing the
        data for this Store. These will have exactly the
        type specified in by 'kind'
        """
        if self._storage is None:
            if self._unbound:
                raise RuntimeError(
                    "Storage of a variable size store cannot be retrieved "
                    "until it is passed to an operation"
                )
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
                if self._parent.scalar:
                    self._storage = self._parent.storage
                else:
                    tiling = Tiling(
                        self._runtime, self.shape, Shape((1,) * self.ndim)
                    )
                    self._storage = self._get_tile(tiling)

        return self._storage

    def __eq__(self, other):
        if not isinstance(other, Store):
            return False
        return (
            self._shape == other._shape
            and self._dtype == other._dtype
            and self._scalar == other._scalar
            and self._transform == other._transform
            and self._parent == other._parent
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self._shape,
                self._dtype,
                self._scalar,
                self._transform,
                self._parent,
            )
        )

    def get_root(self):
        if self._parent is None:
            return self
        else:
            return self._parent.get_root()

    def set_storage(self, storage, shape=None):
        assert isinstance(storage, RegionField) or isinstance(storage, Future)
        self._storage = storage
        self._unbound = False
        if shape is not None:
            self._shape = shape

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
                f"kind: {self.kind.__name__}, storage: {self._storage})>"
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
            storage=self._storage if self._scalar else None,
            optimize_scalar=self._scalar,
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
            shape,
            self._dtype,
            storage=self._storage if self._scalar else None,
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

        transform = Shift(self._runtime, dim, -start)
        shape = self._shape.update(dim, stop - start)
        if self._shape == shape:
            return self
        return Store(
            self._runtime,
            shape,
            self._dtype,
            storage=self._storage if self._scalar else None,
            optimize_scalar=self._scalar,
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

        transform = Transpose(self._runtime, axes)
        shape = transform.compute_shape(self._shape)
        return Store(
            self._runtime,
            shape,
            self._dtype,
            storage=self._storage if self._scalar else None,
            optimize_scalar=self._scalar,
            parent=self,
            transform=transform,
        )

    def delinearize(self, dim, shape):
        transform = Delinearize(self._runtime, dim, shape)
        shape = transform.compute_shape(self._shape)
        return Store(
            self._runtime,
            shape,
            self._dtype,
            storage=self._storage if self._scalar else None,
            optimize_scalar=self._scalar,
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

    def get_numpy_array(self, context=None):
        transform = self.get_inverse_transform()
        return self.storage.get_numpy_array(
            self.shape, context=context, transform=transform
        )

    def _serialize_transform(self, launcher):
        if self._parent is not None:
            self._transform.serialize(launcher)
            self._parent._serialize_transform(launcher)
        else:
            launcher.add_scalar_arg(-1, ty.int32)

    def serialize(self, launcher):
        launcher.add_scalar_arg(self._scalar, bool)
        launcher.add_scalar_arg(self.ndim, ty.int32)
        launcher.add_scalar_arg(self._dtype.code, ty.int32)
        self._serialize_transform(launcher)

    def find_key_partition(self):
        if self._scalar:
            return NoPartition()
        launch_shape = self._partition_manager.compute_launch_shape(self)
        if launch_shape is None:
            return NoPartition()
        else:
            tile_shape = self._partition_manager.compute_tile_shape(
                self.shape, launch_shape
            )
            return Tiling(self._runtime, tile_shape, launch_shape)

    def find_or_create_partition(self, functor):
        assert not self.scalar
        if functor in self._partitions:
            return self._partitions[functor]

        transform = self.get_inverse_transform()
        part = functor.construct(
            self.storage.region, self.shape, inverse_transform=transform
        )
        self._partitions[functor] = part
        return part
