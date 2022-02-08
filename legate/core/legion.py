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

from __future__ import absolute_import, division, print_function

import os
import struct  # For packing and unpacking C data into/out-of futures
import sys
import weakref

import numpy as np

from legion_cffi import ffi, lib as legion

from .types import _Dtype

assert "LEGATE_MAX_DIM" in os.environ
LEGATE_MAX_DIM = int(os.environ["LEGATE_MAX_DIM"])
assert "LEGATE_MAX_FIELDS" in os.environ
LEGATE_MAX_FIELDS = int(os.environ["LEGATE_MAX_FIELDS"])

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


# Helper method for python 3 support
def _itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()


# We can't call out to the CFFI from inside of finalizer methods
# because that can risk a deadlock (CFFI's lock is stupid, they
# take it still in python so if a garbage collection is triggered
# while holding it you can end up deadlocking trying to do another
# CFFI call inside a finalizer because the lock is not reentrant).
# Therefore we defer deletions until we end up launching things
# later at which point we know that it is safe to issue deletions
_pending_unordered = dict()
# We also have some deletion operations which are only safe to
# be done if we know the Legion runtime is still running so we'll
# move them here and only issue the when we know we are inside
# of the execution of a task in some way
_pending_deletions = list()


def legate_task_preamble(runtime, context):
    """
    This function sets up internal Legate state for a task in Python.
    In general, users only need to worry about calling this function
    at the beginning of sub-tasks on the Python side. The Legate
    Core will perform the necessary call to this function for the
    top-level task.
    """
    assert context not in _pending_unordered
    _pending_unordered[context] = list()


def legate_task_progress(runtime, context):
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


def legate_task_postamble(runtime, context):
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
def dispatch(func):
    def launch(launcher, runtime, context, *args):
        # This context should always be in the dictionary
        legate_task_progress(runtime, context)
        return func(launcher, runtime, context, *args)

    return launch


class Point(object):
    def __init__(self, p=None, dim=None):
        """
        The Point class wraps a `legion_domain_point_t` in the Legion C API.
        """
        if dim is None:
            self.point = legion.legion_domain_point_origin(0)
        else:
            self.point = legion.legion_domain_point_origin(dim)
        if p is not None:
            self.set_point(p)

    @property
    def dim(self):
        return self.point.dim

    def __getitem__(self, key):
        if key >= self.dim:
            raise KeyError("key cannot exceed dimensionality")
        return self.point.point_data[key]

    def __setitem__(self, key, value):
        if key >= self.dim:
            raise KeyError("key cannot exceed dimensionality")
        self.point.point_data[key] = value

    def __hash__(self):
        value = hash(self.dim)
        for idx in range(self.dim):
            value = value ^ hash(self[idx])
        return value

    def __eq__(self, other):
        if self.dim != len(other):
            return False
        for idx in range(self.dim):
            if self[idx] != other[idx]:
                return False
        return True

    def __len__(self):
        return self.dim

    def __iter__(self):
        for idx in range(self.dim):
            yield self[idx]

    def __repr__(self):
        p_strs = [str(self[i]) for i in range(self.dim)]
        return "Point(p=[" + ",".join(p_strs) + "])"

    def __str__(self):
        p_strs = [str(self[i]) for i in range(self.dim)]
        return "<" + ",".join(p_strs) + ">"

    def set_point(self, p):
        try:
            if ffi.typeof(p).cname == "legion_domain_point_t":
                ffi.addressof(self.point)[0] = p
                return
        except TypeError:
            pass
        try:
            if len(p) > LEGATE_MAX_DIM:
                raise ValueError(
                    "Point cannot exceed "
                    + str(LEGATE_MAX_DIM)
                    + " dimensions set from LEGATE_MAX_DIM"
                )
            self.point.dim = len(p)
            for i, x in enumerate(p):
                self.point.point_data[i] = x
        except TypeError:
            self.point.dim = 1
            self.point.point_data[0] = p

    def raw(self):
        return self.point


class Rect(object):
    def __init__(self, hi=None, lo=None, exclusive=True, dim=None):
        """
        The Rect class represents an N-D rectangle of dense points. It wraps a
        dense `legion_domain_t` (this is a special case for Domains; in the
        general case a Domain can also contain a sparsity map).
        """
        self._lo = Point(dim=dim)
        self._hi = Point(dim=dim)
        if dim is None:
            self.rect = legion.legion_domain_empty(0)
        else:
            self.rect = legion.legion_domain_empty(dim)
        if hi:
            self.set_bounds(lo=lo, hi=hi, exclusive=exclusive)
        elif lo is not None:
            raise ValueError("'lo' cannot be set without 'hi'")

    @property
    def lo(self):
        return self._lo

    @property
    def hi(self):
        return self._hi

    @property
    def dim(self):
        assert self._lo.dim == self._hi.dim
        return self._lo.dim

    def get_volume(self):
        volume = 1
        for i in range(self.dim):
            volume *= self.hi[i] - self.lo[i] + 1
        return volume

    def __eq__(self, other):
        try:
            if self.lo != other.lo:
                return False
            if self.hi != other.hi:
                return False
        except AttributeError:
            return False
        return True

    def __hash__(self):
        result = hash(self.dim)
        for idx in range(self.dim):
            result = result ^ hash(self.lo[idx])
            result = result ^ hash(self.hi[idx])
        return result

    def __iter__(self):
        p = Point(self._lo)
        dim = self.dim
        yield Point(p)
        while True:
            for idx in range(dim - 1, -2, -1):
                if idx < 0:
                    return
                if p[idx] < self._hi[idx]:
                    p[idx] += 1
                    yield Point(p)
                    break
                p[idx] = self._lo[idx]

    def __repr__(self):
        return f"Rect(lo={repr(self._lo)},hi={repr(self._hi)},exclusive=False)"

    def __str__(self):
        return str(self._lo) + ".." + str(self._hi)

    def set_bounds(self, lo, hi, exclusive=True):
        if len(hi) > LEGATE_MAX_DIM:
            raise ValueError(
                "Point cannot exceed "
                + str(LEGATE_MAX_DIM)
                + " dimensions set from LEGATE_MAX_DIM"
            )
        if exclusive:
            self._hi.set_point([x - 1 for x in hi])
        else:
            self._hi.set_point(hi)
        if lo is not None:
            if len(lo) != len(hi):
                raise ValueError("Length of 'lo' must equal length of 'hi'")
            self._lo.set_point(lo)
        else:
            self._lo.set_point((0,) * len(hi))

    def raw(self):
        dim = self._hi.dim
        self.rect.dim = dim
        for i in range(dim):
            self.rect.rect_data[i] = self._lo[i]
            self.rect.rect_data[dim + i] = self._hi[i]
        return self.rect


class Domain(object):
    def __init__(self, domain):
        """
        The Domain class wraps a `legion_domain_t` in the Legion C API. A
        Domain is the value stored by an IndexSpace. It consists of an N-D
        rectangle describing an upper bound on the points contained in the
        IndexSpace as well as optional sparsity map describing the actual
        non-dense set of points. If there is no sparsity map then the domain
        is purely the set of dense points represented by the rectangle bounds.

        Note that Domain objects do not copy the contents of the provided
        `legion_domain_t` handle, nor do they take ownership of if. It is up
        to the calling code to ensure that the memory backing the original
        handle will not be collected while this object is in use.
        """

        self.domain = domain
        self.rect = Rect(dim=domain.dim)
        for i in range(domain.dim):
            self.rect.lo[i] = self.domain.rect_data[i]
            self.rect.hi[i] = self.domain.rect_data[domain.dim + i]
        self.dense = legion.legion_domain_is_dense(domain)

    @property
    def dim(self):
        return self.rect.dim

    def get_volume(self):
        return legion.legion_domain_get_volume(self.domain)

    def get_rects(self):
        # NOTE: For debugging only!
        create = getattr(
            legion,
            f"legion_rect_in_domain_iterator_create_{self.dim}d",
        )
        destroy = getattr(
            legion,
            f"legion_rect_in_domain_iterator_destroy_{self.dim}d",
        )
        valid = getattr(
            legion,
            f"legion_rect_in_domain_iterator_valid_{self.dim}d",
        )
        step = getattr(
            legion,
            f"legion_rect_in_domain_iterator_step_{self.dim}d",
        )
        get_rect = getattr(
            legion,
            f"legion_rect_in_domain_iterator_get_rect_{self.dim}d",
        )
        rects = []
        iterator = create(self.domain)
        while valid(iterator):
            nd_rect = get_rect(iterator)
            lo = [nd_rect.lo.x[i] for i in range(self.dim)]
            hi = [nd_rect.hi.x[i] for i in range(self.dim)]
            rects.append(Rect(hi=hi, lo=lo, exclusive=False, dim=self.dim))
            step(iterator)
        destroy(iterator)
        return rects


class Transform(object):
    def __init__(self, M, N, eye=True):
        """
        A Transform wraps an `legion_transform_{m}x{n}_t` in the Legion C API.
        A transform is simply an MxN matrix that can be used to convert Point
        objects from one coordinate space to another.
        """

        self.M = M
        self.N = N
        if eye:
            self.trans = np.eye(M, N, dtype=np.int64)
        else:
            self.trans = np.zeros((M, N), dtype=np.int64)
        self.handle = None

    def apply(self, point):
        """
        Convert an N-D Point into an M-D point using this transform
        """
        if len(point) != self.N:
            raise ValueError("Dimension mismatch")
        result = ()
        for m in xrange(self.M):
            value = 0
            for n in xrange(self.N):
                value += self.trans[m, n] * point[n]
            result += (value,)
        return result

    def compose(self, outer):
        """
        Construct a composed transform of this transform with another transform
        """
        if outer.N != self.M:
            raise ValueError("Dimension mismatch")
        result = Transform(outer.M, self.N, eye=False)
        np.matmul(outer.trans, self.trans, out=result.trans)
        return result

    def raw(self):
        if self.handle is None:
            self.handle = legion.legion_domain_transform_identity(
                self.M, self.N
            )
        self.handle.m = self.M
        self.handle.n = self.N
        for m in range(self.M):
            for n in range(self.N):
                self.handle.matrix[m * self.N + n] = self.trans[m, n]
        return self.handle

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.M == other.M
            and self.N == other.N
            and np.array_equal(self.trans, other.trans)
        )

    def __hash__(self):
        return hash(self.trans.tobytes())

    def __str__(self):
        return np.array_repr(self.trans).replace("\n", "").replace(" ", "")


# An Affine Transform for points in one space to points in another
class AffineTransform(object):
    def __init__(self, M, N, eye=True):
        """
        An AffineTransform wraps a `legion_affine_transform_{m}x{n}_t` in the
        Legion C API. The AffineTransform class represents an affine transform
        as a MxN affine transform as an (M+1)x(N+1) matrix and can used to
        transform N-D Point objects into M-D Point objects. AffineTransform
        objects can also be naturally composed to construct new
        AffineTransforms.
        """

        self.M = M
        self.N = N
        if eye:
            self.transform = np.eye(M + 1, N + 1, dtype=np.int64)
        else:
            self.transform = np.zeros((M + 1, N + 1), dtype=np.int64)
            self.transform[self.M, self.N] = 1
        self.handle = None

    @property
    def offset(self):
        return self.transform[: self.M, self.N]

    @offset.setter
    def offset(self, offset):
        self.transform[: self.M, self.N] = offset

    @property
    def trans(self):
        return self.transform[: self.M, : self.N]

    @trans.setter
    def trans(self, transform):
        self.transform[: self.M, : self.N] = transform

    def apply(self, point):
        """
        Convert an N-D Point into an M-D point using this transform
        """
        if len(point) != self.N:
            raise ValueError("Dimension mismatch")
        pin = np.ones(self.N + 1, dtype=np.int64)
        pin[: self.N] = point
        pout = np.dot(self.transform, pin)
        return tuple(pout[: self.M])

    def compose(self, outer):
        """
        Construct a composed transform of this transform with another transform
        """
        if outer.N != self.M:
            raise ValueError("Dimension mismatch")
        result = AffineTransform(outer.M, self.N, eye=False)
        np.matmul(outer.transform, self.transform, out=result.transform)
        return result

    def raw(self):
        if self.handle is None:
            self.handle = legion.legion_domain_affine_transform_identity(
                self.M, self.N
            )
        self.handle.transform.m = self.M
        self.handle.transform.n = self.N
        self.handle.offset.dim = self.M
        for m in range(self.M):
            for n in range(self.N):
                self.handle.transform.matrix[m * self.N + n] = self.transform[
                    m, n
                ]
            self.handle.offset.point_data[m] = self.transform[m, self.N]
        return self.handle

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.M == other.M
            and self.N == other.N
            and np.array_equal(self.transform, other.transform)
        )

    def __hash__(self):
        return hash(self.transform.tobytes())

    def __str__(self):
        return np.array_repr(self.transform).replace("\n", "").replace(" ", "")


class IndexSpace(object):
    def __init__(self, context, runtime, handle, parent=None, owned=True):
        """
        An IndexSpace object wraps a `legion_index_space_t` in the Legion C
        API. An IndexSpace provides a name for a collection of (sparse) points.
        An IndexSpace can serve as the names for the rows in a LogicalRegion
        or be used for dispatching collections of tasks or other operations
        using index space launches.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        handle : legion_index_space_t
            Created handle for an index space from a Legion C API call
        parent : IndexPartition
            Parent index partition for this index space if any
        owned : bool
            Whether this object owns the handle for this index space and
            can delete the index space handle when the object is collected
        """
        self.context = context
        self.runtime = runtime
        self.parent = parent
        self.handle = handle
        self.children = None
        self.owned = owned
        self._domain = None
        if owned and self.parent is not None and not self.parent.owned:
            raise ValueError(
                "IndexSpace can only own its handle if the parent "
                "IndexPartition also owns its handle"
            )

    def __del__(self):
        # We only need to delete top-level index spaces
        # Ignore any deletions though that occur after the task is done
        if self.owned and self.parent is None:
            self.destroy(unordered=True)

    def _can_delete(self):
        if not self.owned:
            return False
        if self.parent is not None:
            return self.parent.parent._can_delete()
        # Must be owned at the root to enable deletion
        return self.owned

    def add_child(self, child):
        """
        Add a child partition to this IndexSpace.
        """
        if child.owned and not self.owned:
            raise ValueError(
                "IndexSpace parent must be owned if "
                "IndexPartition children are also owned"
            )
        if self.children is None:
            # Make this a weak set since partitions can be removed
            # independently from their parent index space
            self.children = set()
        self.children.add(child)

    def destroy(self, unordered=False):
        """
        Force deletion of this IndexSpace regardless of ownership
        This must be a root index space

        Parameters
        ----------
        unordered : bool
            Whether this request is coming directly from the task or through
            and unordered channel such as a garbage collection
        """
        if self.parent is not None:
            raise RuntimeError("Only root IndexSpace objects can be destroyed")
        if not self.owned:
            return
        self.owned = False
        # Check to see if we're still inside the context of the task
        # If not then we just need to leak this index space
        if self.context not in _pending_unordered:
            return
        if unordered:
            # See if we have a _logical_handle from an OutputRegion
            # that we need to pass along to keep things alive
            if hasattr(self, "_logical_handle"):
                _pending_unordered[self.context].append(
                    ((self.handle, self._logical_handle), type(self))
                )
            else:
                _pending_unordered[self.context].append(
                    ((self.handle, None), type(self))
                )
        else:
            legion.legion_index_space_destroy_unordered(
                self.runtime, self.context, self.handle, False
            )

    def get_root(self):
        """
        Find the root of IndexSpace tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    @property
    def domain(self):
        """
        Return a Domain that represents the points in this index space
        """
        if self._domain is None:
            self._domain = Domain(
                legion.legion_index_space_get_domain(self.runtime, self.handle)
            )
        return self._domain

    def get_bounds(self):
        """
        Return a Rect that represents the upper bounds of the IndexSpace.
        """
        return self.domain.rect

    def get_volume(self):
        """
        Return the total number of points in the IndexSpace
        """
        return self.domain.get_volume()

    def get_dim(self):
        """
        Return the dimension of the IndexSpace
        """
        if self._domain is not None:
            return self._domain.dim
        return legion.legion_index_space_get_dim(self.handle)


class PartitionFunctor(object):
    """
    PartitionFunctor objects provide a common interface to computing
    IndexPartition objects using Legion's support for dependent partitioning.
    Each kind of dependent partitioning operator in Legion can be accessed
    through a custom PartitionFunctor.
    """

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        """
        The generic interface for computing an IndexPartition

        Parameters
        ----------
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        context : legion_context_t
            The Legion context from get_legion_context()
        parent : IndexSpace
            The parent index space for which we're computing this partition
        color_space : IndexSpace
            The color space defining the names of subspaces to be created
        kind : legion_partition_kind_t
            Description of the disjointness and completeness property
            (if they are known) for the partition
        part_id : int
            Desired name for the color of this partition in the parent
            index space's color space (if any)
        """
        raise NotImplementedError("implement in derived classes")


class PartitionByRestriction(PartitionFunctor):
    def __init__(self, transform, extent):
        """
        PartitionByRestriction constructs a tesselated IndexPartition where an
        IndexSpace is created for each Point in the color space by projecting
        the point through the transform to create the lower bound point and
        then adding the extent. Note that this can be used to create both
        disjoint and aliased partitions, depending on the properties of the
        transform and the extent.
        """

        self.transform = transform
        self.extent = extent

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        return legion.legion_index_partition_create_by_restriction(
            runtime,
            context,
            parent.handle,
            color_space.handle,
            self.transform.raw(),
            self.extent.raw(),
            kind,
            part_id,
        )


class PartitionByImage(PartitionFunctor):
    def __init__(self, region, part, field, mapper=0, tag=0):
        """
        PartitionByImage projects an existing IndexPartition through a field
        of Points that point from one LogicalRegion into an IndexSpace.
        """

        self.region = region
        self.part = part
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        return legion.legion_index_partition_create_by_image(
            runtime,
            context,
            parent.handle,
            self.part.handle,
            self.region.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class PartitionByImageRange(PartitionFunctor):
    def __init__(self, region, part, field, mapper=0, tag=0):
        """
        PartitionByImageRange projects an existing IndexPartition through a
        field of Rects that point from one LogicalRegion into a range of points
        in a destination IndexSpace.
        """

        self.region = region
        self.part = part
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        return legion.legion_index_partition_create_by_image_range(
            runtime,
            context,
            parent.handle,
            self.part.handle,
            self.region.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class PartitionByPreimage(PartitionFunctor):
    def __init__(self, projection, region, parent, field, mapper=0, tag=0):
        """
        Partition by preimage induces a partition on the index space of
        'region' by taking a field of region that points into the index
        partition 'projection" and reversing the mapping.
        """
        self.projection = projection
        self.region = region
        self.parent = parent
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        return legion.legion_index_partition_create_by_preimage(
            runtime,
            context,
            self.projection.handle,
            self.region.handle,
            self.parent.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class PartitionByPreimageRange(PartitionFunctor):
    def __init__(self, projection, region, parent, field, mapper=0, tag=0):
        """
        Partition by preimage range induces a partition on the index space of
        'region' by taking a field of region that points into ranges of the
        index partition 'projection" and reversing the mapping.
        """
        self.projection = projection
        self.region = region
        self.parent = parent
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        return legion.legion_index_partition_create_by_preimage_range(
            runtime,
            context,
            self.projection.handle,
            self.region.handle,
            self.parent.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class EqualPartition(PartitionFunctor):
    """
    EqualPartition will construct an IndexPartition that creates IndexSpace
    children with roughly equal numbers of points in each child.
    """

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        return legion.legion_index_partition_create_equal(
            runtime,
            context,
            parent.handle,
            color_space.handle,
            1,
            part_id,
        )


class PartitionByWeights(PartitionFunctor):
    def __init__(self, weights):
        """
        PartitionByWeights will construct an IndexPartition with the number of
        points in each child IndexSpace being allocated proportionally to the
        the relative weights.
        """

        self.weights = weights

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        if isinstance(self.weights, FutureMap):
            return legion.legion_index_partition_create_by_weights_future_map(
                runtime,
                context,
                parent.handle,
                self.weights.handle,
                color_space.handle,
                1,
                part_id,
            )
        elif isinstance(self.weights, list):
            num_weights = len(self.weights)
            colors = ffi.new("legion_domain_point_t[%d]" % num_weights)
            weights = ffi.new("int[%d]" % num_weights)
            for i in xrange(num_weights):
                colors[i] = Point([i]).raw()
                weights[i] = self.weights[i]
            return legion.legion_index_partition_create_by_weights(
                runtime,
                context,
                parent.handle,
                colors,
                weights,
                num_weights,
                color_space.handle,
                1,
                part_id,
            )
        else:
            raise TypeError("Unsupported type for PartitionByWeights")


class PartitionByDomain(PartitionFunctor):
    def __init__(self, domains):
        """
        PartitionByDomain will construct an IndexPartition given an explicit
        mapping of colors to domains.

        Parameters
        ----------
        domains : FutureMap | dict[Point, Rect]
        """
        self.domains = domains

    def partition(self, runtime, context, parent, color_space, kind, part_id):
        if isinstance(self.domains, FutureMap):
            return legion.legion_index_partition_create_by_domain_future_map(
                runtime,
                context,
                parent.handle,
                self.domains.handle,
                color_space.handle,
                True,  # perform_intersections
                kind,
                part_id,
            )
        elif isinstance(self.domains, dict):
            num_domains = len(self.domains)
            assert num_domains <= color_space.get_volume()
            colors = ffi.new("legion_domain_point_t[%d]" % num_domains)
            domains = ffi.new("legion_domain_t[%d]" % num_domains)
            for (i, (point, rect)) in enumerate(self.domains.items()):
                colors[i] = point.raw()
                domains[i] = rect.raw()
            return legion.legion_index_partition_create_by_domain(
                runtime,
                context,
                parent.handle,
                colors,
                domains,
                num_domains,
                color_space.handle,
                True,  # perform_intersections
                kind,
                part_id,
            )
        else:
            raise TypeError("Unsupported type for PartitionByDomain")


# TODO more kinds of partition functors here


class IndexPartition(object):
    def __init__(
        self,
        context,
        runtime,
        parent,
        color_space,
        functor=None,
        handle=None,
        kind=legion.LEGION_COMPUTE_KIND,
        part_id=legion.legion_auto_generate_id(),
        owned=True,
        keep=False,
    ):
        """
        An IndexPartition wraps a `legion_index_partition_t` in the Legion C
        API. It describes a partitioning of an IndexSpace into a collection of
        child IndexSpace objects. IndexPartition objects can be both disjoint
        and aliased. They can also be both complete and incomplete partitions.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        parent : IndexSpace
            The parent index space for this partition
        color_space : IndexSpace
            The index space that describes this subregions in this partition
        functor : PartitionFunctor
            Any object that implements the PartitionFunctor interface
        handle : legion_index_partition_t
            Created handle for an index partition from a Legion C API call
        kind : legion_partition_kind_t
            The disjointness and completeness properties for the partition
        part_id : int
            The color for the partition in parent index space
        owned : bool
            Whether this object owns the handle for this index space and
            can delete the index space handle when the object is collected
        keep : bool
            Whether to keep a reference to the functor object that
            was used to create this index partition
        """
        self.context = context
        self.runtime = runtime
        self.parent = parent
        self.color_space = color_space
        if handle is None:
            if functor is None:
                raise ValueError(
                    "'functor' must be " "specified if 'handle is None"
                )
            if not isinstance(functor, PartitionFunctor):
                raise TypeError("'functor' must be a 'PartitionFunctor'")
            handle = functor.partition(
                runtime, context, parent, color_space, kind, part_id
            )
            if keep:
                self.functor = functor
            else:
                self.functor = None
        elif functor is not None:
            raise ValueError("'functor' must be None if 'handle' is specified")
        else:
            self.functor = None
        self.handle = handle
        self.children = dict()
        self.owned = owned
        if owned and not self.parent.owned:
            raise ValueError(
                "IndexPartition can only own its handle if "
                "the parent IndexSpace also owns its handle"
            )
        self.parent.add_child(self)

    def __del__(self):
        # Record a pending deletion if this task is still executing
        if self.owned and self.parent._can_delete():
            self.destroy(unordered=True, recursive=True)

    def get_child(self, point):
        """
        Find the child IndexSpace assocated with the point in the color space.
        """
        if point in self.children:
            return self.children[point]
        child_handle = legion.legion_index_partition_get_index_subspace_domain_point(  # noqa: E501
            self.runtime, self.handle, point.raw()
        )
        child = IndexSpace(
            self.context,
            self.runtime,
            handle=child_handle,
            parent=self,
            owned=self.owned,
        )
        self.children[point] = child
        return child

    def destroy(self, unordered=False, recursive=True):
        """
        Force deletion of this IndexPartition regardless of ownership

        Parameters
        ----------
        unordered : bool
            Whether this request is coming directly from the task or through
            and unordered channel such as a garbage collection
        recursive : bool
            Whether to recursively destroy down the index space tree
        """
        if not self.owned:
            return
        self.owned = False
        # Check to see if we're still inside the context of the task
        # If not then we just need to leak this index partition
        if self.context not in _pending_unordered:
            return
        if unordered:
            # See if we have a _logical_handle from an OutputRegion
            # that we need to pass along to keep things alive
            if hasattr(self, "_logical_handle"):
                _pending_unordered[self.context].append(
                    (
                        (self.handle, recursive, self._logical_handle),
                        type(self),
                    )
                )
            else:
                _pending_unordered[self.context].append(
                    ((self.handle, recursive, None), type(self))
                )
        else:
            legion.legion_index_partition_destroy_unordered(
                self.runtime, self.context, self.handle, False, recursive
            )
        if recursive:
            for child in _itervalues(self.children):
                child.owned = False

    def get_root(self):
        """
        Return the root IndexSpace in this tree.
        """
        return self.parent.get_root()


class FieldSpace(object):
    def __init__(self, context, runtime, handle=None, owned=True):
        """
        A FieldSpace wraps a `legion_field_space_t` in the Legion C API.
        It is used to represent the columns in a LogicalRegion. Users can
        allocate and destroy fields in a field space dynamically.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        handle : legion_field_space_t
            Created handle for a field space from a Legion C API call; if
            provided then this object cannot be used to allocate new fields
        owned : bool
            Whether this object owns the handle for this field space and
            can delete the field space when the object is collected
        """
        self.context = context
        self.runtime = runtime
        if handle is None:
            self.handle = legion.legion_field_space_create(runtime, context)
            self.alloc = legion.legion_field_allocator_create(
                runtime, context, self.handle
            )
        else:
            self.handle = handle
            self.alloc = None
        self.fields = dict()
        self.owned = owned

    def __del__(self):
        # Only delete this if the task is still executing otherwise we leak it
        if self.owned:
            self.destroy(unordered=True)

    @property
    def has_space(self):
        return len(self.fields) < LEGATE_MAX_FIELDS

    def allocate_field(
        self, size_or_type, field_id=legion.legion_auto_generate_id()
    ):
        """
        Allocate a field in the field space by its size or by inferring its
        size from some type representation that is passed in
        """
        if self.alloc is None:
            raise TypeError("Field allocations not allowed on this object")
        if field_id in self.fields:
            raise ValueError("'field_id' must be unique in an index space")
        if len(self.fields) == LEGATE_MAX_FIELDS:
            raise RuntimeError(
                "Exceeded maximum number of fields ("
                + str(LEGATE_MAX_FIELDS)
                + " in field space"
            )
        if isinstance(size_or_type, _Dtype):
            return self.allocate_field_dtype(size_or_type, field_id=field_id)
        elif isinstance(size_or_type, Future):
            return self.allocate_field_from_future(
                size_or_type, field_id=field_id
            )
        elif isinstance(size_or_type, int):
            field_id = legion.legion_field_allocator_allocate_field(
                self.alloc, size_or_type, field_id
            )
            self.fields[field_id] = size_or_type
            return field_id
        else:
            return self.allocate_field_ctype(size_or_type, field_id=field_id)

    def allocate_field_ctype(
        self, ctype, field_id=legion.legion_auto_generate_id()
    ):
        """
        Allocate a field in the field space based on the ctypes type.
        """
        if self.alloc is None:
            raise TypeError("Field allocations not allowed on this object")
        if field_id in self.fields:
            raise ValueError("'field_id' must be unique in an index space")
        if len(self.fields) == LEGATE_MAX_FIELDS:
            raise RuntimeError(
                "Exceeded maximum number of fields ("
                + str(LEGATE_MAX_FIELDS)
                + " in field space"
            )
        field_id = legion.legion_field_allocator_allocate_field(
            self.alloc, ffi.sizeof(ctype), field_id
        )
        self.fields[field_id] = ctype
        return field_id

    def allocate_field_dtype(
        self, dtype, field_id=legion.legion_auto_generate_id()
    ):
        """
        Allocate a field in the field space based on the NumPy dtype.
        """
        if self.alloc is None:
            raise TypeError("Field allocations not allowed on this object")
        if field_id in self.fields:
            raise ValueError("'field_id' must be unique in an index space")
        if len(self.fields) == LEGATE_MAX_FIELDS:
            raise RuntimeError(
                "Exceeded maximum number of fields ("
                + str(LEGATE_MAX_FIELDS)
                + " in field space"
            )
        field_id = legion.legion_field_allocator_allocate_field(
            self.alloc, dtype.size, field_id
        )
        self.fields[field_id] = dtype
        return field_id

    def allocate_field_from_future(
        self, future, field_id=legion.legion_auto_generate_id()
    ):
        """
        Allocate a field based on a size stored in a Future
        """
        if self.alloc is None:
            raise TypeError("Field allocations not allowed on this object")
        if field_id in self.fields:
            raise ValueError("'field_id' must be unique in an index space")
        if len(self.fields) == LEGATE_MAX_FIELDS:
            raise RuntimeError(
                "Exceeded maximum number of fields ("
                + str(LEGATE_MAX_FIELDS)
                + " in field space"
            )
        field_id = legion.legion_field_allocator_allocate_field_future(
            self.alloc, future.handle, field_id
        )
        self.fields[field_id] = future
        return field_id

    def destroy_field(self, field_id, unordered=False):
        """
        Destroy a field in the field space and reclaim its resources.
        Set `unordered` to `True` if this is done inside a garbage collection.
        """
        if self.alloc is None:
            raise TypeError("Field allocations not allowed on this object")
        if not self.owned:
            return
        if field_id not in self.fields:
            raise ValueError("Destroyed field not contained in field space")
        # Check to see if we're still in the context of the task
        # If not then we just have to leak this field
        if self.context not in _pending_unordered:
            return
        if unordered:
            _pending_unordered[self.context].append(
                ((self.alloc, field_id), FieldID)
            )
        else:
            legion.legion_field_allocator_free_field(self.alloc, field_id)
            del self.fields[field_id]

    def get_type(self, field_id):
        """
        Return the type of the object used to create the field.
        """
        return self.fields[field_id]

    def __len__(self):
        return len(self.fields)

    def destroy(self, unordered=False):
        """
        Force deletion of this FieldSpace regardless of ownership

        Parameters
        ----------
        unordered : bool
            Whether this request is coming directly from the task or through
            and unordered channel such as a garbage collection
        """
        if not self.owned:
            return
        self.owned = False
        # Check to see if we're still in the context of the task
        # If not then we just have to leak this field space
        if self.context not in _pending_unordered:
            return
        if unordered:
            _pending_unordered[self.context].append(
                ((self.handle, self.alloc), type(self))
            )
        else:
            legion.legion_field_space_destroy_unordered(
                self.runtime, self.context, self.handle, False
            )
            if self.alloc is not None:
                legion.legion_field_allocator_destroy(self.alloc)


class FieldID(object):
    def __init__(self, field_space, fid, type):
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

    def destroy(self, unordered=False):
        """
        Deallocate this field from the field space
        """
        self.field_space.destroy_field(self.field_id, unordered)

    @property
    def fid(self):
        return self.field_id

    @property
    def type(self):
        return self._type


class Region(object):
    def __init__(
        self,
        context,
        runtime,
        index_space,
        field_space,
        handle=None,
        parent=None,
        owned=True,
    ):
        """
        A Region wraps a `legion_logical_region_t` in the Legion C API.
        A logical region describes a dataframe-like representation of program
        data with the rows described by an IndexSpace and the columns described
        by a FieldSpace.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        index_space : IndexSpace
            The index space for this logical region
        field_space : FieldSpace
            The field space for this logical region
        handle : legion_logical_region_t
            Created handle for a logical region from a Legion C API call
        parent : Partition
            Parent logical partition for this logical region, if any
        owned : bool
            Whether this object owns the handle for this region and
            can delete the region when the object is collected
        """
        if (
            parent is not None
            and index_space.parent is not parent.index_partition
        ):
            raise ValueError("Illegal index space for parent")
        self.context = context
        self.runtime = runtime
        self.index_space = index_space
        self.field_space = field_space
        self.parent = parent
        if handle is None:
            if parent is None:
                # Create a new logical region
                handle = legion.logical_region_create(
                    runtime,
                    context,
                    index_space.handle,
                    field_space.handle,
                    True,
                )
            else:
                handle = legion.legion_logical_partition_get_subregion(
                    runtime, parent.handle, index_space.handle
                )
        self.handle = handle
        self.owned = owned
        # Make this a WeakValueDicitionary so that entries can be
        # removed once the partitions are deleted
        self.children = weakref.WeakValueDictionary()

    def __del__(self):
        if self.owned and self.parent is None:
            self.destroy(unordered=True)

    def same_handle(self, other):
        return (
            type(self) == type(other)
            and self.handle.tree_id == other.handle.tree_id
            and self.handle.index_space.id == other.handle.index_space.id
            and self.handle.field_space.id == other.handle.field_space.id
        )

    def __str__(self):
        return (
            f"Region("
            f"tid: {self.handle.tree_id}, "
            f"is: {self.handle.index_space.id}, "
            f"fs: {self.handle.field_space.id})"
        )

    def destroy(self, unordered=False):
        """
        Force deletion of this Region regardless of ownership

        Parameters
        ----------
        unordered : bool
            Whether this request is coming directly from the task or through
            and unordered channel such as a garbage collection
        """
        if self.parent is not None:
            raise RuntimeError("Only root Region objects can be destroyed")
        if not self.owned:
            return
        self.owned = False
        # Check to see if we're still in the context of our task
        # If not we need to leak this region
        if self.context not in _pending_unordered:
            return
        if unordered:
            _pending_unordered[self.context].append((self.handle, type(self)))
        else:
            legion.legion_logical_region_destroy_unordered(
                self.runtime, self.context, self.handle, False
            )

    def get_child(self, index_partition):
        """
        Find the Partition object that corresponds to the corresponding
        IndexPartition object for the IndexSpace of this Region.
        """
        if index_partition in self.children:
            return self.children[index_partition]
        child = Partition(self.context, self.runtime, index_partition, self)
        self.children[index_partition] = child
        return child

    def get_root(self):
        """
        Return the root Region in this tree.
        """
        if not self.parent:
            return self
        return self.parent.parent.get_root()


class Partition(object):
    def __init__(self, context, runtime, index_partition, parent, handle=None):
        """
        A Partition wraps a `legion_logical_partition_t` in the Legion C API.

        Parameters
        ----------
        context : legion_context_t
            The Legion context from get_legion_context()
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        index_partition : IndexPartition
            The index partition associated with this logical partition
        parent : Region
            Parent logical region for this logical partition if any
        handle : legion_logical_partition_t
            Created handle for logical partition from a Legion C API call
        """
        if index_partition.parent is not parent.index_space:
            raise ValueError("Illegal index partition for parent")
        self.context = context
        self.runtime = runtime
        self.index_partition = index_partition
        self.parent = parent
        if handle is None:
            handle = legion.legion_logical_partition_create(
                runtime,
                context,
                parent.handle,
                index_partition.handle,
            )
        self.handle = handle
        self.children = dict()

    @property
    def color_space(self):
        return self.index_partition.color_space

    def destroy(self):
        """
        This method is deprecated and is a no-op
        Partition objects never need to explicitly destroy their handles
        """
        pass

    def get_child(self, point):
        """
        Return the child Region associated with the point in the color space.
        """
        if point in self.children:
            return self.children[point]
        child_space = self.index_partition.get_child(point)
        handle = legion.legion_logical_partition_get_logical_subregion(
            self.runtime, self.handle, child_space.handle
        )
        child = Region(
            self.context,
            self.runtime,
            child_space,
            self.parent.field_space,
            handle=handle,
            parent=self,
        )
        self.children[point] = child
        return child

    def get_root(self):
        """
        Return the Region at the root of this region tree.
        """
        return self.parent.get_root()


class Fill(object):
    def __init__(self, region, parent, field, future, mapper=0, tag=0):
        """
        A Fill object provides a mechanism for launching fill operations
        in the Legion C API. A fill operation initializes the data in a
        field of a logical region to a specific value. Importantly this is
        performed lazily so fills incur near-zero cost.

        Parameters
        ----------
        region : Region
            The logical region to be filled
        parent : Region
            The parent logical region where priviles are derived from
        field : FieldID or int
            The name of the field to be filled
        future : Future
            Future describing the value to use for performing the fill
        mapper : int
            ID of the mapper to use for mapping the fill
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        if not isinstance(field, FieldID) and not isinstance(field, int):
            raise TypeError("Fill field must be int or FieldID")
        self.launcher = legion.legion_fill_launcher_create_from_future(
            region.handle,
            parent.handle if parent is not None else parent.get_root().handle,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            future.handle,
            legion.legion_predicate_true(),
            mapper,
            tag,
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_fill_launcher_destroy
        )

    def set_point(self, point):
        """
        Set the point to describe this fill for sharding
        with control replication

        Parameters
        ----------
        point : Point or legion_domain_point_t
            The point value to associate with this fill
        """
        if isinstance(point, Point):
            legion.legion_fill_launcher_set_point(self.launcher, point.raw())
        else:
            legion.legion_fill_launcher_set_point(self.launcher, point)

    def set_sharding_space(self, space):
        """
        Set the sharding space for this individual fill launch

        Parameters
        ----------
        space : IndexSpace
            The index space for use when performing sharding of this fill
        """
        legion.legion_fill_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the fill to the runtime
        """
        legion.legion_fill_launcher_execute(runtime, context, self.launcher)


class IndexFill(object):
    def __init__(
        self,
        partition,
        proj,
        parent,
        field,
        future,
        mapper=0,
        tag=0,
        space=None,
    ):
        """
        An IndexFill object provides a mechanism for launching index space fill
        operations in the Legion C API. Index fill operations enable many
        subregions in a sub-tree to be filled concurrently and efficiently.  An
        index fill operation initializes the data in the field of each of the
        logical regions targeted by the index fill operation. Importantly, this
        is performed lazily so index fills incur near-zero cost.

        partition : Partition
            The logical partition upper bound from which to project the
            point fills from the index fill
        proj : int
            Projection function ID to describe how to project
            from the upper bound partition
        parent : Region
            The parent logical region where privileges are derived from
        field : FieldID or int
            The name of the field to be filled
        future: Future
            Future describing the value to use for performing the index fill
        mapper : int
            ID of the mapper to use for mapping the index fill
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        space : IndexSpace or Domain (default:None)
            launch space domain
        """
        if not isinstance(field, FieldID) and not isinstance(field, int):
            raise TypeError("Fill field must be int or FieldID")
        if (
            space is not None
            and not isinstance(space, IndexSpace)
            and not isinstance(space, Domain)
        ):
            raise TypeError(
                "IndexFill launch space must be IndexSpace or Domain"
            )
        if space is None:
            self.launcher = legion.legion_index_fill_launcher_create_from_future_with_space(  # noqa: E501
                partition.index_partition.color_space.handle,
                partition.handle,
                parent.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                future.handle,
                proj,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        elif isinstance(self.space, IndexSpace):
            self.launcher = legion.legion_index_fill_launcher_create_from_future_with_space(  # noqa: E501
                space.handle,
                partition.handle,
                parent.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                future.handle,
                proj,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        else:
            self.launcher = legion.legion_index_fill_launcher_create_from_future_with_domain(  # noqa: E501
                space.domain,
                partition.handle,
                parent.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                future.handle,
                proj,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_fill_launcher_destroy
        )

    def set_sharding_space(self, space):
        """
        Set the sharding space to use for this index fill launch
        with control replication

        Parameters
        ----------
        space : IndexSpace
            The index space to use as the sharding space
        """
        legion.legion_index_fill_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the index fill to the runtime
        """
        legion.legion_index_fill_launcher_execute(
            runtime, context, self.launcher
        )


class Copy(object):
    def __init__(self, mapper=0, tag=0):
        """
        A Copy object provides a mechanism for launching explicit
        region-to-region copy operations. Note: you should NOT use
        these for trying to move data between memories! Copy launchers
        should only be used for logically moving data between fields.

        Parameters
        ----------
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_copy_launcher_create(
            legion.legion_predicate_true(), mapper, tag
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_copy_launcher_destroy
        )
        self.src_req_index = 0
        self.dst_req_index = 0

    def set_possible_src_indirect_out_of_range(self, flag):
        """
        For gather indirection copies indicate whether any of the
        source indirection pointers may point out of bounds in the
        source logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether source pointers may point outside the privileges
        """
        legion.legion_copy_launcher_set_possible_src_indirect_out_of_range(
            self._launcher, flag
        )

    def set_possible_dst_indirect_out_of_range(self, flag):
        """
        For scatter indirection copies indicate whether any of the
        destination indirection pointers may point out of bounds in the
        destination logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether destination pointers may point outside
            the privileges
        """
        legion.legion_copy_launcher_set_possible_dst_indirect_out_of_range(
            self._launcher, flag
        )

    def add_src_requirement(
        self,
        region,
        fields,
        parent=None,
        tag=0,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a source region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a source of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as sources of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        coherence : int
            The coherence mode for which to access this region
        """
        legion.legion_copy_launcher_add_src_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_READ_ONLY,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_copy_launcher_add_src_field(
                self.launcher,
                self.src_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.src_req_index += 1

    def add_dst_requirement(
        self,
        region,
        fields,
        parent=None,
        tag=0,
        redop=0,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a destination region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a destination of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as destinations of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        redop : int
            Optional reduction operator ID to reduce to the destination fields
        coherence : int
            The coherence mode for which to access this region
        """
        if redop == 0:
            legion.legion_copy_launcher_add_dst_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                region.handle,
                legion.LEGION_WRITE_DISCARD,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                tag,
                False,
            )
        else:
            legion.legion_copy_launcher_add_dst_region_requirement_logical_region_reduction(  # noqa: E501
                self.launcher,
                region.handle,
                redop,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                tag,
                False,
            )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_copy_launcher_add_dst_field(
                self.launcher,
                self.dst_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.dst_req_index += 1

    def add_src_indirect_requirement(
        self,
        region,
        field,
        parent=None,
        tag=0,
        is_range=False,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a source indirection region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a source indirection of the copy
        field : int or FieldID
            The ID of the field to serve as a source indirection of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        legion.legion_copy_launcher_add_src_indirect_region_requirement_logical_region(  # noqa: E501
            self.launcher,
            region.handle,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            is_range,
            False,
        )

    def add_dst_indirect_requirement(
        self,
        region,
        field,
        parent=None,
        tag=0,
        is_range=False,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a destination indirection region requirement to the copy operation

        Parameters
        ----------
        region : Region
            The logical region to serve as a destination
            indirection of the copy
        field : int or FieldID
            The ID of the field to serve as a destination
            indirection field of the copy
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        legion.legion_copy_launcher_add_dst_indirect_region_requirement_logical_region(  # noqa: E501
            self.launcher,
            region.handle,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            is_range,
            False,
        )

    def set_point(self, point):
        """
        Set the point to describe this copy for sharding
        with control replication

        Parameters
        ----------
        point : Point or legion_domain_point_t
            The point value to associate with this copy
        """
        if isinstance(point, Point):
            legion.legion_copy_launcher_set_point(self.launcher, point.raw())
        else:
            legion.legion_copy_launcher_set_point(self.launcher, point)

    def set_sharding_space(self, space):
        """
        Set the sharding space for this individual copy launch

        Parameters
        ----------
        space : IndexSpace
            The index space for use when performing sharding of this copy
        """
        legion.legion_copy_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the copy operation to the runtime
        """
        if self.src_req_index != self.dst_req_index:
            raise RuntimeError(
                "Number of source and destination requirements "
                + "must match for copies"
            )
        legion.legion_copy_launcher_execute(runtime, context, self.launcher)


class IndexCopy(object):
    def __init__(self, domain, mapper=0, tag=0):
        """
        An IndexCopy object provides a mechanism for launching explicit
        region-to-region copies between many different subregions
        simultaneously.  Note: you should NOT use these for trying to move data
        between memories!  Copy launchers should only be used for logically
        moving data between different fields.

        Parameters
        ----------
        domain : Domain
            The domain of points for the index space launch
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_index_copy_launcher_create(
            domain.raw(), legion.legion_predicate_true(), mapper, tag
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_copy_launcher_destroy
        )
        self.src_req_index = 0
        self.dst_req_index = 0

    def set_possible_src_indirect_out_of_range(self, flag):
        """
        For gather indirection copies indicate whether any of the
        source indirection pointers may point out of bounds in the
        source logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether source pointers may point outside the privileges
        """
        legion.legion_index_copy_launcher_set_possible_src_indirect_out_of_range(  # noqa: E501
            self._launcher, flag
        )

    def set_possible_dst_indirect_out_of_range(self, flag):
        """
        For scatter indirection copies indicate whether any of the
        destination indirection pointers may point out of bounds in the
        destination logical regions.

        Parameters
        ----------
        flag : bool
            Indicate whether destination pointers may point outside
            the privileges
        """
        legion.legion_index_copy_launcher_set_possible_dst_indirect_out_of_range(  # noqa: E501
            self._launcher, flag
        )

    def add_src_requirement(
        self,
        upper_bound,
        fields,
        projection,
        parent=None,
        tag=0,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a source region requirement to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            source of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as sources of the copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_copy_launcher_add_src_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_ONLY,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_copy_launcher_add_src_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_ONLY,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_copy_launcher_add_src_field(
                self.launcher,
                self.src_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.src_req_index += 1

    def add_dst_requirement(
        self,
        upper_bound,
        fields,
        projection,
        parent=None,
        tag=0,
        redop=0,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a destination region requirement to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            source of the copy
        fields : int or FieldID or List[int] or List[FieldID]
            The ID(s) of the fields to serve as sources of the copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        redop : int
            Optional reduction operator ID to reduce the destination fields
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            if redop == 0:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_region(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    legion.LEGION_WRITE_DISCARD,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
            else:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    redop,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
        elif isinstance(upper_bound, Partition):
            if redop == 0:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_partition(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    legion.LEGION_WRITE_DISCARD,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
            else:
                legion.legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction(  # noqa: E501
                    self.launcher,
                    upper_bound.handle,
                    projection,
                    redop,
                    coherence,
                    upper_bound.get_root().handle
                    if parent is None
                    else parent.handle,
                    tag,
                    False,
                )
        else:
            raise TypeError("'upper_bound' must be a Region or a Partition")
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_copy_launcher_add_dst_field(
                self.launcher,
                self.dst_req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        self.dst_req_index += 1

    def add_src_indirect_requirement(
        self,
        upper_bound,
        field,
        projection,
        parent=None,
        tag=0,
        is_range=False,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a source indirection region requirement to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            source of the index copy
        field : int or FieldID
            The ID of the field to serve as a source indirection of
            the index copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_copy_launcher_add_src_indirect_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_copy_launcher_add_src_indirect_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")

    def add_dst_indirect_requirement(
        self,
        upper_bound,
        field,
        projection,
        parent=None,
        tag=0,
        is_range=False,
        coherence=legion.LEGION_EXCLUSIVE,
        **kwargs,
    ):
        """
        Add a destination indirection region requirement
        to the index copy operation

        Parameters
        ----------
        upper_bound: Region or Partition
            The upper bound logical region or partition to serve as a
            destination of the index copy
        field : int or FieldID
            The ID of the field to serve as a source indirection of
            the index copy
        projection : int
            The ID of a projection function to compute specific subregions for
            each point in the index space launch domain
        parent : Region
            The parent logical region from which privileges are derived
        tag : int
            A mapping tag to pass to the mapper for context of this requirement
        is_range : bool
            Indicate whether the indirection field is a single pointer or
            a range (e.g. Rect) of points
        coherence : int
            The coherence mode for which to access this region
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_region(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                is_range,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")

    def set_sharding_space(self, space):
        """
        Set the sharding space to use for this index copy launch
        with control replication

        Parameters
        ----------
        space : IndexSpace
            The index space to use as the sharding space
        """
        legion.legion_index_copy_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    @dispatch
    def launch(self, runtime, context):
        if self.src_req_index != self.dst_req_index:
            raise RuntimeError(
                "Number of source and destination requirements "
                + "must match for copies"
            )
        legion.legion_index_copy_launcher_execute(
            runtime, context, self.launcher
        )


class Attach(object):
    def __init__(
        self,
        region,
        field,
        data,
        mapper=0,
        tag=0,
        read_only=False,
    ):
        """
        An Attach object provides a mechanism for attaching external data to
        a logical region, thereby allowing Legion to use external data in
        place for performing computations.

        Parameters
        ----------
        region : Region
            The logical region to which external data will be attached
        field : int or FieldID
            The field ID to which the data will be attached
        data : memoryview
            Input data in a memoryview
        mapper : int
            ID of the mapper to use for mapping the operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        read_only : bool
            Whether this buffer should only be accessed read-only
        """
        self.launcher = legion.legion_attach_launcher_create(
            region.handle, region.handle, legion.LEGION_EXTERNAL_INSTANCE
        )
        self.region = region
        self._launcher = ffi.gc(
            self.launcher, legion.legion_attach_launcher_destroy
        )
        legion.legion_attach_launcher_add_cpu_soa_field(
            self.launcher,
            ffi.cast(
                "legion_field_id_t",
                field.fid if isinstance(field, FieldID) else field,
            ),
            ffi.from_buffer(data),
            data.f_contiguous,
        )

    def set_restricted(self, restricted):
        """
        Set whether restricted coherence should be used on the logical region.
        If restricted coherence is enabled, changes to the data in the logical
        region will be eagerly reflected back to the external buffer.
        """
        legion.legion_attach_launcher_set_restricted(self.launcher, restricted)

    def set_mapped(self, mapped):
        """
        Set whether the resulting PhysicalRegion should be considered mapped
        in the enclosing task context.
        """
        legion.legion_attach_launcher_set_mapped(self.launcher, mapped)

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the attach operation to the runtime

        Returns
        -------
        PhysicalRegion that names the attached external resource
        """
        return PhysicalRegion(
            legion.legion_attach_launcher_execute(
                runtime, context, self.launcher
            ),
            self.region,
        )


class Detach(object):
    def __init__(self, region, flush=True):
        """
        A Detach operation will unbind an external resource from a logical
        region.  This will also allow any outstanding mutations to the logical
        region to be flushed back to the external memory allocation.

        Parameters
        ----------
        region : PhysicalRegion
            The physical region describing an external resource to be detached
        flush : bool
            Whether to flush changes to the logical region to the
            external allocation
        """
        self.physical_region = region
        # Keep a reference to the logical region to ensure that
        # it is not deleted before this detach operation can run
        self.region = region.region
        self.flush = flush

    def launch(self, runtime, context, unordered=False):
        """
        Dispatch the detach operation to the runtime

        Returns
        -------
        Future containing no data that completes when detach operation is done
        If 'unordered' is set to true then you must call legate_task_progress
        before using the returned Future
        """
        # Check to see if we're still inside the context of the task
        # If not then we just need to leak this detach because it can't be done
        if context not in _pending_unordered:
            return None
        if unordered:
            future = Future()
            _pending_unordered[context].append(((self, future), type(self)))
            return future
        else:
            return Future(
                legion.legion_unordered_detach_external_resource(
                    runtime,
                    context,
                    self.physical_region.handle,
                    self.flush,
                    unordered,
                )
            )


class ExternalResources(object):
    def __init__(self, handle):
        """
        Stores a collection of physical regions that were attached together
        using the same IndexAttach operation. Wraps a
        `legion_external_resources_t` object from the Legion C API.
        """
        self.handle = handle

    def __del__(self):
        self.destroy(unordered=True)

    def destroy(self, unordered):
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


class IndexAttach(object):
    def __init__(
        self,
        parent,
        field,
        shard_local_data,
        mapper=0,
        tag=0,
    ):
        """
        A variant of Attach that allows attaching multiple pieces of external
        data as sub-regions of the same parent region. Each piece may reside
        on a different address space.

        Parameters
        ----------
        parent : Region
            The parent region to which external data will be attached as
            sub-regions
        field : int | FieldID
            The field ID to which the data will be attached
        shard_local_data : dict[Region, memoryview]
            Maps sub-regions to buffers on the shard's local address space.
            Each sub-region will be attached to the corresponding buffer.
            Each shard should pass a set of distinct subregions, and all
            sub-regions must be disjoint.
        mapper : int
            ID of the mapper to use for mapping the operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_index_attach_launcher_create(
            parent.handle,
            legion.LEGION_EXTERNAL_INSTANCE,
            True,  # restricted
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_attach_launcher_destroy
        )
        fields = ffi.new("legion_field_id_t[1]")
        fields[0] = field.fid if isinstance(field, FieldID) else field
        # Find a local system memory
        machine = legion.legion_machine_create()
        query = legion.legion_memory_query_create(machine)
        legion.legion_memory_query_only_kind(query, legion.SYSTEM_MEM)
        legion.legion_memory_query_local_address_space(query)
        assert legion.legion_memory_query_count(query) > 0
        mem = legion.legion_memory_query_first(query)
        legion.legion_memory_query_destroy(query)
        legion.legion_machine_destroy(machine)
        for (sub_region, buf) in shard_local_data.items():
            assert sub_region.parent.parent is parent
            legion.legion_index_attach_launcher_attach_array_soa(
                self.launcher,
                sub_region.handle,
                ffi.from_buffer(buf),
                buf.f_contiguous,
                fields,
                1,  # num_fields
                mem,
            )

    def set_restricted(self, restricted):
        """
        Set whether restricted coherence should be used on the logical region.
        If restricted coherence is enabled, changes to the data in the logical
        region will be eagerly reflected back to the external buffers.
        """
        legion.legion_index_attach_launcher_set_restricted(
            self.launcher, restricted
        )

    def set_deduplicate_across_shards(self, deduplicate):
        """
        Set whether the runtime should check for duplicate resources
        """
        legion.legion_index_attach_launcher_set_deduplicate_across_shards(
            self.launcher, deduplicate
        )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the operation to the runtime

        Returns
        -------
        An ExternalResources object that names the attached resources
        """
        return ExternalResources(
            legion.legion_attach_external_resources(
                runtime, context, self.launcher
            )
        )


class IndexDetach(object):
    def __init__(self, external_resources, flush=True):
        """
        An IndexDetach operation will unbind a collection of external resources
        that were attached together to a logical region through an IndexAttach.
        This will also allow any outstanding mutations to the logical region to
        be flushed back to the external memory allocations.

        Parameters
        ----------
        external_resources : ExternalResources
            The external resources to be detached
        flush : bool
            Whether to flush changes to the logical region to the external
            allocations
        """
        self.external_resources = external_resources
        self.flush = flush

    def launch(self, runtime, context):
        """
        Dispatch the operation to the runtime

        Returns
        -------
        Future containing no data that completes when the operation is done
        """
        return Future(
            legion.legion_detach_external_resources(
                runtime,
                context,
                self.external_resources.handle,
                self.flush,
                False,  # unordered
            )
        )


class Acquire(object):
    def __init__(self, region, fields, mapper=0, tag=0):
        """
        An Acquire operation provides a mechanism for temporarily relaxing
        restricted coherence on a logical region, thereby enabling Legion
        to manage coherence with multiple copies of the data.

        Parameters
        ----------
        region : Region
            The logical region on which to relax restricted coherence
        fields : int or FieldID or List[int] or List[FieldID]
            The fields to perform the attach on
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_acquire_launcher_create(
            region.handle,
            region.handle,
            legion.legion_predicate_true(),
            mapper,
            tag,
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_acquire_launcher_destroy
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_acquire_launcher_add_field(
                self.launcher,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
            )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the acquire operation to the runtime
        """
        legion.legion_acquire_launcher_execute(runtime, context, self.launcher)


class Release(object):
    def __init__(self, region, fields, mapper=0, tag=0):
        """
        A Release operation will undo any acquire operations by putting
        restricted coherence requirements back onto a logical region.

        Parameters
        ----------
        region : Region
            The logical region on which to relax restricted coherence
        fields : int or FieldID or List[int] or List[FieldID]
            The fields to perform the attach on
        mapper : int
            ID of the mapper to use for mapping the copy operation
        tag : int
            Tag to pass to the mapper to provide context for any mapper calls
        """
        self.launcher = legion.legion_release_launcher_create(
            region.handle,
            region.handle,
            legion.legion_predicate_true(),
            mapper,
            tag,
        )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_release_launcher_destroy
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_release_launcher_add_field(
                self.launcher,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
            )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the release operation to the runtime
        """
        legion.legion_release_launcher_execute(runtime, context, self.launcher)


class Future(object):
    def __init__(self, handle=None, type=None):
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

    def __del__(self):
        self.destroy(unordered=True)

    # We cannot use this as __eq__ because then we would have to define a
    # compatible __hash__, which would not be sound because self.handle can
    # change during the lifetime of a Future object, and thus so would the
    # object's hash. So we just leave the default `f1 == f2 <==> f1 is f2`.
    def same_handle(self, other):
        return type(self) == type(other) and self.handle == other.handle

    def __str__(self):
        return f"Future({str(self.handle.impl)[16:-1]})"

    def destroy(self, unordered):
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

    def set_value(self, runtime, data, size, type=None):
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

    def get_buffer(self, size=None):
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

    def get_size(self):
        """
        Return the size of the buffer that the future stores.
        This will block until the future completes if it has not already.
        """
        return legion.legion_future_get_untyped_size(self.handle)

    def get_string(self):
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

    def is_ready(self, subscribe=False):
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

    def wait(self):
        """
        Block waiting for the future to complete
        """
        legion.legion_future_get_void_result(self.handle)

    @property
    def type(self):
        return self._type


class OutputRegion(object):
    def __init__(
        self,
        context,
        runtime,
        field_space=None,
        fields=[],
        global_indexing=True,
        existing=None,
        flags=None,
        proj=None,
        parent=None,
        coherence=legion.LEGION_EXCLUSIVE,
        tag=0,
    ):
        """
        An OutputRegion creates a name for a logical region that will be
        produced as an output from executing a/an (index) task. The bounds of
        the index space for this region will not be immediately known, but
        users can still use the names of the logical region and index space
        throughout the program without needing to block to wait for the size to
        resolve.

        Parameters
        ----------
        context : legion_context_t
            Context for the enclosing parent task
        runtime : legion_runtime_t
            Handle for the Legion runtime
        field_space : FieldSpace
            The field space to use for the creation of a new logical region
        fields : int or FieldID or List[int] or List[FieldID]
            The fields that will be created in the output region
        global_indexing : bool
            Whether or not the runtime should use a global indexing approach
            to compute the new index space for the output logical region.
            If set to true, the output space will be a 1-D space with a
            prefix sum over all the subregions. If set to false then the
            output index space will be 2-D reflecting the output size for
            each of the different point tasks contributing.
        existing : Region or Partition
            An optional existing region or partition for which the output
            will be a new instance
        flags : int
            Flags to attach to the region requirement for the output region
        proj : int
            The ID of a projection function to use for projection
        parent : Region
            The parent logical region from which privileges are derived
        coherence : int
            The coherence mode for accessing the logical region
        tag : int
            The tag to attach to the region requirement to provide context
        """
        self.context = context
        self.runtime = runtime
        self.fields = set()

        if field_space is not None:
            if existing is not None:
                raise ValueError(
                    "'existing' cannot be set if 'field_space' is"
                )
            self.field_space = field_space
            self.region = None
            self.partition = None
            self.handle = legion.legion_output_requirement_create(
                field_space.handle, ffi.NULL, 0, global_indexing
            )
        elif existing is not None:
            if isinstance(existing, Region):
                self.field_space = None
                self.region = existing
                self.partition = None
                if proj is None:
                    req = (
                        legion.legion_region_requirement_create_logical_region(
                            existing.handle,
                            legion.LEGION_WRITE_DISCARD,
                            coherence,
                            existing.get_root().handle
                            if parent is None
                            else parent.handle,
                            tag,
                            False,
                        )
                    )
                else:
                    req = legion.legion_region_requirement_create_logical_region_projection(  # noqa: E501
                        existing.handle,
                        proj,
                        legion.LEGION_WRITE_DISCARD,
                        coherence,
                        existing.get_root().handle
                        if parent is None
                        else parent.handle,
                        tag,
                        False,
                    )
                if flags is not None:
                    legion.legion_region_requirement_add_flags(req, flags)
                self.handle = (
                    legion.legion_output_requirement_create_region_requirement(
                        req
                    )
                )
                legion.legion_region_requirement_destroy(req)
            elif isinstance(existing, Partition):
                self.field_space = None
                self.region = existing.parent
                self.partition = existing
                if proj is None:
                    raise ValueError(
                        "'proj' must be set for partition outputs"
                    )
                req = (
                    legion.legion_region_requirement_create_logical_partition(
                        existing.handle,
                        proj,
                        legion.LEGION_WRITE_DISCARD,
                        coherence,
                        existing.get_root().handle
                        if parent is None
                        else parent.handle,
                        tag,
                        False,
                    )
                )
                if flags is not None:
                    legion.legion_region_requirement_add_flags(req, flags)
                self.handle = (
                    legion.legion_output_requirement_create_region_requirement(
                        req
                    )
                )
                legion.legion_region_requirement_destroy(req)
            else:
                raise TypeError("'existing' must be a Region or Partition")
        else:
            raise ValueError(
                "'existing' must be set if 'field_space' is not set"
            )

        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            self.add_field(field)

    def __del__(self):
        self.destroy(unordered=True)

    def destroy(self, unordered):
        """
        Eagerly destroy this OutputRegion before the garbage collector does
        It is illegal to use the OutputRegion after this call

        Parameters
        ----------
        unordered : bool
            Whether this OutputRegion is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_future_destroy(self.handle)
        self.handle = None

    def add_field(self, field, instance=True):
        """
        Add a field to this output region

        Parameters
        ----------
        field : int or FieldID
            field to add to the output region
        instance : bool
            whether this should be added to the instance vector of
            the region requirement for the output region
        """
        if field not in self.fields:
            self.fields.add(field)
            legion.legion_output_requirement_add_field(
                self.handle,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                instance,
            )

    def get_region(self, owned=True):
        """
        Parameters
        ----------
        owned : bool
            Whether to take ownership of the created
            IndexSpace and Region objects

        Returns
        -------
        New logical region created by the output region
        """
        if self.region is None:
            assert self.field_space is not None
            handle = legion.legion_output_requirement_get_parent(self.handle)
            index_space = IndexSpace(
                self.context, self.runtime, handle.index_space, owned=owned
            )
            # Dangle a reference here to make sure we don't accidentally
            # cleanup the handle before the IndexSpace goes away
            index_space._logical_handle = handle
            self.region = Region(
                self.context,
                self.runtime,
                index_space,
                self.field_space,
                handle,
                owned=owned,
            )
        return self.region

    def get_partition(self, owned=True):
        """
        Parameters
        ----------
        owned : bool
            Whether to take ownership of the created IndexPartition object

        Returns
        -------
        New logical partition created by the output region
        """
        if self.partition is None:
            assert self.field_space is not None
            context = self.context
            runtime = self.runtime
            parent = self.get_region()
            partition_handle = legion.legion_output_requirement_get_partition(
                self.handle
            )
            color_space = IndexSpace(
                context,
                runtime,
                legion.legion_index_partition_get_color_space(
                    runtime,
                    partition_handle.index_partition,
                ),
                owned=False,
            )
            index_partition = IndexPartition(
                context,
                runtime,
                parent.index_space,
                color_space,
                handle=partition_handle.index_partition,
                owned=owned,
            )
            # Careful here, need to dangle a reference to the partition_handle
            # off the index_partition because its handle references it
            index_partition._logical_handle = partition_handle
            self.partition = Partition(
                context, runtime, index_partition, parent, partition_handle
            )
        return self.partition


class PhysicalRegion(object):
    def __init__(self, handle, region):
        """
        A PhysicalRegion object represents an actual mapping of a logical
        region to a physical allocation in memory and its associated layout.
        PhysicalRegion objects can be both mapped and unmapped. A mapped
        PhysicalRegion contains the most recent valid copy of the logical
        region's data, whereas an unmapped PhysicalRegion can contains a
        stale copy of the data.

        Parameters
        ----------
        handle : legion_physical_region_t
            The handle for a physical region that this object will own
        region : Region
            The logical region for this physical region
        """
        self.handle = handle
        self.region = region

    def __del__(self):
        self.destroy(unordered=True)

    def destroy(self, unordered):
        """
        Eagerly destroy this PhysicalRegion before the garbage collector does
        It is illegal to use the PhysicalRegion after this call

        Parameters
        ----------
        unordered : bool
            Whether this PhysicalRegion is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_physical_region_destroy(self.handle)
        self.handle = None

    def is_mapped(self):
        """
        Returns
        -------
        bool indicating if this PhysicalRegion is currently mapped
        """
        # It's possible that this method is invoked on an already GC'd
        # PhysicalRegion object, due to the Python GC not being topologically
        # ordered. We ignore the call in this case.
        if self.handle is None:
            return False
        return legion.legion_physical_region_is_mapped(self.handle)

    def wait_until_valid(self):
        """
        Block waiting until the data in this physical region
        to be ready to access
        """
        legion.legion_physical_region_wait_until_valid(self.handle)

    @dispatch
    def remap(self, runtime, context):
        """
        Remap this physical region so that it contains a valid copy of the
        data for the logical region that it represents
        """
        legion.legion_runtime_remap_region(runtime, context, self.handle)

    # Launching one of these means remapping it
    def launch(self, runtime, context):
        self.remap(runtime, context)

    def unmap(self, runtime, context, unordered=False):
        """
        Unmap this physical region from the current logical region
        If 'unordered=True' you must call legate_task_progress to
        guarantee that this unmapping is finished
        """
        # It's possible that this method is invoked on an already GC'd
        # PhysicalRegion object, due to the Python GC not being topologically
        # ordered. We ignore the call in this case.
        if self.handle is None:
            return
        # Check to see if we're still inside the context of the task
        # If not then we just ignore this unmap, it will be done anyway
        if context not in _pending_unordered:
            return
        if unordered:
            _pending_unordered[context].append((self, type(self)))
        else:
            legion.legion_runtime_unmap_region(runtime, context, self.handle)


class InlineMapping(object):
    def __init__(
        self,
        region,
        fields,
        read_only=False,
        mapper=0,
        tag=0,
        parent=None,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        An InlineMapping object provides a mechanism for creating a mapped
        PhysicalRegion of a logical region for the local task to directly
        access the data in the logical region. Note that inline mappings
        do block deferred execution and therefore they should be used
        primarily as a productivity feature for loading and storing data
        infrequently. They should never be used in performance critical code.

        Parameters
        ----------
        region : Region
            The logical region to map
        fields : int or FieldID or List[int] or List[FieldID]
            The fields of the logical region to map
        read_only : bool
            Whether the inline mapping will only be reading the data
        mapper : int
            ID of the mapper for managing the mapping of the inline mapping
        tag : int
            Tag to pass to the mapper to provide calling context
        parent : Region
            Parent logical region from which privileges are derived
        coherence : int
            The coherence mode for the inline mapping
        """
        if read_only:
            self.launcher = legion.legion_inline_launcher_create_logical_region(  # noqa: E501
                region.handle,
                legion.LEGION_READ_ONLY,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                0,
                False,
                mapper,
                tag,
            )
        else:
            self.launcher = legion.legion_inline_launcher_create_logical_region(  # noqa: E501
                region.handle,
                legion.LEGION_READ_WRITE,
                coherence,
                region.get_root().handle if parent is None else parent.handle,
                0,
                False,
                mapper,
                tag,
            )
        self.region = region
        self._launcher = ffi.gc(
            self.launcher, legion.legion_inline_launcher_destroy
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_inline_launcher_add_field(
                self.launcher,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the inline mapping to the runtime

        Returns
        -------
        PhysicalRegion
        """
        return PhysicalRegion(
            legion.legion_inline_launcher_execute(
                runtime, context, self.launcher
            ),
            self.region,
        )


class Task(object):
    def __init__(self, task_id, data=None, size=0, mapper=0, tag=0):
        """
        A Task object provides a mechanism for launching individual sub-tasks
        from the current parent task. For scalability we encourage the use of
        IndexTasks in the common case, but there are certainly many common
        uses of individual task launches as well.

        task_id : int
            ID of the task to launch
        data : object that implements the Python buffer protocol
            The data to pass to the task as a buffer by-value
        size : int
            Number of bytes of data to pass as a by-value argument
        mapper : int
            ID of the mapper for managing the mapping of the task
        tag : int
            Tag to pass to the mapper to provide calling context
        """
        if data:
            if size <= 0:
                raise ValueError("'size' must be positive")
            self.launcher = legion.legion_task_launcher_create_from_buffer(
                task_id,
                ffi.from_buffer(data),
                size,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
            # Hold a reference to the data to prevent collection
            self.data = data
        else:
            if size != 0:
                raise ValueError("'size' must be zero if there is no 'data'")
            self.launcher = legion.legion_task_launcher_create_from_buffer(
                task_id,
                ffi.NULL,
                size,
                legion.legion_predicate_true(),
                mapper,
                tag,
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_task_launcher_destroy
        )
        self.req_index = 0
        self.outputs = []

    def add_no_access_requirement(
        self,
        region,
        fields,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a no-access region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_NO_ACCESS,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_requirement(
        self,
        region,
        fields,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a read-only region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_READ_ONLY,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_write_requirement(
        self,
        region,
        fields,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a write-discard region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_WRITE_DISCARD,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_write_requirement(
        self,
        region,
        fields,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a read-write region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region(
            self.launcher,
            region.handle,
            legion.LEGION_READ_WRITE,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_reduction_requirement(
        self,
        region,
        fields,
        redop,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a reduction region requirement to the task

        Parameters
        ----------
        region : Region
            The logical region for the region requirement
        fields : int or FieldID or List[int] or List[FieldID]
            List of field identifiers for the region requirement
        redop : int
            ReductionOp ID that will be used in the sub-task
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        legion.legion_task_launcher_add_region_requirement_logical_region_reduction(  # noqa: E501
            self.launcher,
            region.handle,
            redop,
            coherence,
            region.get_root().handle if parent is None else parent.handle,
            tag,
            False,
        )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_task_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_task_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_future(self, future):
        """
        Record a future as a precondition on running this task

        Parameters
        ----------
        future : Future
            The future to record as a precondition
        """
        legion.legion_task_launcher_add_future(self.launcher, future.handle)

    def add_output(self, output):
        """
        Add an output region to the region requirements for this task

        Parameters
        ----------
        output : OutputRegion
            The output region that will be determined by this index launch
        """
        self.outputs.append(output)

    def add_outputs(self, outputs):
        """
        Add a output regions to the region requirements for this task

        Parameters
        ----------
        outputs : List[OutputRegion]
            The output regions that will be determined by this index launch
        """
        self.outputs.extend(outputs)

    def set_point(self, point):
        """
        Set the point to describe this task for sharding
        with control replication

        Parameters
        ----------
        point : Point or legion_domain_point_t
            The point value to associate with this task
        """
        if isinstance(point, Point):
            legion.legion_task_launcher_set_point(self.launcher, point.raw())
        else:
            legion.legion_task_launcher_set_point(self.launcher, point)

    def set_sharding_space(self, space):
        """
        Set the sharding space for this individual task launch

        Parameters
        ----------
        space : IndexSpace
            The index space for use when performing sharding of this task
        """
        legion.legion_task_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    def set_local_function(self, local):
        """
        Set a flag indicating whether this task can be considered a local
        function task. Specifically that means it only has future
        preconditions with no region arguments.

        Parameters
        ----------
        local : bool
            Whether we can treat the task as a local function task
        """
        legion.legion_task_launcher_set_local_function_task(
            self.launcher, local
        )

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch the task launch to the runtime

        Returns
        -------
        Future that will complete when the task is done and carries
        the return value of the task if any
        """
        num_outputs = len(self.outputs)
        if num_outputs == 0:
            return Future(
                legion.legion_task_launcher_execute(
                    runtime, context, self.launcher
                )
            )
        else:
            outputs = ffi.new("legion_output_requirement_t[%d]" % num_outputs)
            for i, output in enumerate(self.outputs):
                outputs[i] = output.handle
            return Future(
                legion.legion_task_launcher_execute_outputs(
                    runtime,
                    context,
                    self.launcher,
                    outputs,
                    len(self.outputs),
                )
            )


class FutureMap(object):
    def __init__(self, handle=None):
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

    def __del__(self):
        self.destroy(unordered=True)

    def destroy(self, unordered):
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

    def wait(self):
        """
        Wait for all the futures in the future map to complete
        """
        legion.legion_future_map_wait_all_results(self.handle)

    def get_future(self, point):
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
        self, context, runtime, redop, deterministic=False, mapper=0, tag=0
    ):
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
    def from_list(cls, context, runtime, futures):
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
        for i in xrange(num_futures):
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
    def from_dict(cls, context, runtime, domain, futures, collective=False):
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


class IndexTask(object):
    def __init__(
        self, task_id, domain, argmap=None, data=None, size=0, mapper=0, tag=0
    ):
        """
        An IndexTask object provides a mechnanism for launching a collection
        of tasks simultaneously as described by the points in an index space.
        The point tasks inside the index task launch can still name arbitrary
        subsets of data (although if they are interfering, the projection
        functions used must be capable of describing the dependences).

        task_id : int
            ID of the task to launch
        domain : Rect or IndexSpace
            The domain description of the tasks to make (one task per point)
        argmap : ArgumentMap
            Optional argument map for passing data to point tasks
        data : object that implements the Python buffer protocol
            Buffer of byte arguments to pass to all the point tasks
        size : int
            The number of bytes in the data buffer to pass
        mapper : int
            ID of the mapper for managing the mapping of the task
        tag : int
            Tag to pass to the mapper to provide calling context
        """
        if argmap is not None:
            self.argmap = None
        else:
            self.argmap = legion.legion_argument_map_create()
            self._argmap = ffi.gc(
                self.argmap, legion.legion_argument_map_destroy
            )
            argmap = self.argmap
        if isinstance(domain, Rect):
            domain = domain.raw()
        if data:
            if size <= 0:
                raise ValueError("'size' must be positive for 'data'")
            self.launcher = legion.legion_index_launcher_create_from_buffer(
                task_id,
                domain,
                ffi.from_buffer(data),
                size,
                argmap,
                legion.legion_predicate_true(),
                False,
                mapper,
                tag,
            )
            # Hold a reference to the data to prevent collection
            self.data = data
        else:
            if size != 0:
                raise ValueError("'size' must be zero if there is no 'data'")
            self.launcher = legion.legion_index_launcher_create_from_buffer(
                task_id,
                domain,
                ffi.NULL,
                size,
                argmap,
                legion.legion_predicate_true(),
                False,
                mapper,
                tag,
            )
        self._launcher = ffi.gc(
            self.launcher, legion.legion_index_launcher_destroy
        )
        self.req_index = 0
        self.point_args = None
        self.outputs = []

    def add_no_access_requirement(
        self,
        upper_bound,
        fields,
        projection,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a region requirement without any access privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_NO_ACCESS,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_NO_ACCESS,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_requirement(
        self,
        upper_bound,
        fields,
        projection,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a region requirement read-only access privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_ONLY,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_ONLY,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_write_requirement(
        self,
        upper_bound,
        fields,
        projection,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a region requirement with write-discard privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_WRITE_DISCARD,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_WRITE_DISCARD,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_read_write_requirement(
        self,
        upper_bound,
        fields,
        projection,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a region requirement with read-write privileges

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region(
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_WRITE,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                legion.LEGION_READ_WRITE,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_reduction_requirement(
        self,
        upper_bound,
        fields,
        redop,
        projection,
        parent=None,
        tag=0,
        flags=0,
        coherence=legion.LEGION_EXCLUSIVE,
    ):
        """
        Add a region requirement with reduction privileges for a reduction op

        Parameters
        ----------
        upper_bound : Region or Partition
            The upper node in the region tree from which point task
            region requirements will be projected
        fields : int or FieldID or List[int] or List[FieldID]
            The fields for the region requirement
        redop : int
            ID for a reduction operator the tasks will use
        projection : int
            ID for the projection function to use for performing projections
        parent : Region
            The logical region from which privileges are derived
        tag : int
            Tag to pass to the mapper to provide calling context
        flags : int
            Flags to attach to the region requirement
        coherence : int
            The coherence mode for the region requirement
        """
        if isinstance(upper_bound, Region):
            legion.legion_index_launcher_add_region_requirement_logical_region_reduction(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                redop,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        elif isinstance(upper_bound, Partition):
            legion.legion_index_launcher_add_region_requirement_logical_partition_reduction(  # noqa: E501
                self.launcher,
                upper_bound.handle,
                projection,
                redop,
                coherence,
                upper_bound.get_root().handle
                if parent is None
                else parent.handle,
                tag,
                False,
            )
        else:
            raise TypeError("'upper_bound' must be a Region or Partition")
        if not isinstance(fields, list):
            fields = [fields]
        for field in fields:
            legion.legion_index_launcher_add_field(
                self.launcher,
                self.req_index,
                ffi.cast(
                    "legion_field_id_t",
                    field.fid if isinstance(field, FieldID) else field,
                ),
                True,
            )
        if flags > 0:
            legion.legion_index_launcher_add_flags(
                self.launcher, self.req_index, flags
            )
        self.req_index += 1

    def add_future(self, future):
        """
        Add a future precondition to all the points in the index space launch

        Parameters
        ----------
        future : Future
            A future that will be passed as a precondition to all point tasks
        """
        legion.legion_index_launcher_add_future(self.launcher, future.handle)

    def add_point_future(self, argmap):
        """
        Add an additional argument map for passing arguments to each point
        task, each of these arguments will be appended to list of futures
        for each of the point tasks.

        Parameters
        ----------
        argmap : ArgumentMap
            Per-point arguments to be added to the list of future preconditions
        """
        legion.legion_index_launcher_add_point_future(
            self.launcher, argmap.handle
        )

    def add_output(self, output):
        """
        Add an output region to the region requirements for this task

        Parameters
        ----------
        output : OutputRegion
            The output region that will be determined by this index launch
        """
        self.outputs.append(output)

    def add_outputs(self, outputs):
        """
        Add a output regions to the region requirements for this task

        Parameters
        ----------
        outputs : List[OutputRegion]
            The output regions that will be determined by this index launch
        """
        self.outputs.extend(outputs)

    def set_point(self, point, data, size):
        """
        Set the point argument in the argument map for the index task

        Parameters
        ----------
        point : Point
            The point to set in the argument map
        data : object that implements the Python buffer protocol
            The data to be set for the point
        size : int
            The size of the data in the buffer
        """
        if data is None:
            raise ValueError("'data' must be provided to set a point")
        if size <= 0:
            raise ValueError("A positive 'size' must be specified for 'data'")
        point_arg = ffi.new("legion_task_argument_t *")
        point_arg[0].args = ffi.from_buffer(data)
        point_arg[0].arglen = size
        legion.legion_argument_map_set_point(
            self.argmap, point.raw(), point_arg, True
        )
        if not self.point_args:
            self.point_args = list()
        self.point_args.append(point_arg)

    def set_sharding_space(self, space):
        """
        Set the sharding space to use for this index launch
        with control replication

        Parameters
        ----------
        space : IndexSpace
            The index space to use as the sharding space
        """
        legion.legion_index_launcher_set_sharding_space(
            self.launcher, space.handle
        )

    @dispatch
    def launch(self, runtime, context, redop=0):
        """
        Launch this index space task to the runtime

        Parameters
        ----------
        runtime : legion_runtime_t
            Handle to the Legion runtime
        context : legion_context_t
            Handle to the enclosing parent context
        redop : int
            ID of a reduction operation to use for reducing the
            outputs of all the point tasks down to a single value

        Returns
        -------
        FutureMap if redop==0 else Future
        """
        if redop == 0:
            num_outputs = len(self.outputs)
            if num_outputs == 0:
                return FutureMap(
                    legion.legion_index_launcher_execute(
                        runtime, context, self.launcher
                    )
                )
            else:
                outputs = ffi.new(
                    "legion_output_requirement_t[%d]" % num_outputs
                )
                for i, output in enumerate(self.outputs):
                    outputs[i] = output.handle
                return FutureMap(
                    legion.legion_index_launcher_execute_outputs(
                        runtime,
                        context,
                        self.launcher,
                        outputs,
                        num_outputs,
                    )
                )

        else:
            num_outputs = len(self.outputs)
            if num_outputs == 0:
                return Future(
                    legion.legion_index_launcher_execute_deterministic_reduction(  # noqa: E501
                        runtime, context, self.launcher, redop, True
                    )
                )
            else:
                outputs = ffi.new(
                    "legion_output_requirement_t[%d]" % num_outputs
                )
                for i, output in enumerate(self.outputs):
                    outputs[i] = output.handle
                return FutureMap(
                    legion.legion_index_launcher_execute_reduction_and_outputs(
                        runtime,
                        context,
                        self.launcher,
                        redop,
                        True,
                        outputs,
                        num_outputs,
                    )
                )


class Fence(object):
    def __init__(self, mapping=False):
        """
        A Fence operation provides a mechanism for inserting either
        mapping or execution fences into the stream of tasks and
        other operations generated by a program. A mapping fence
        will prevent reordering of operations during the mapping
        process, but has no bearing on execution. An execution fence
        will ensure all tasks and operations that come perform the
        fence are ordered with respect to all the tasks and
        operations that come after it.

        Parameters
        ----------
        mapping : bool
            Whether this is a mapping fence or not
        """
        self.mapping = mapping

    @dispatch
    def launch(self, runtime, context):
        """
        Dispatch this fence to the runtime
        """
        if self.mapping:
            return Future(
                legion.legion_runtime_issue_mapping_fence(runtime, context)
            )
        else:
            return Future(
                legion.legion_runtime_issue_execution_fence(runtime, context)
            )


class ArgumentMap(object):
    def __init__(self, handle=None, future_map=None):
        """
        An ArgumentMap is a object that allows for the passing of
        data directly to individual point tasks in IndexTask launches.

        handle : legion_argument_map_t
            The handle of which to take ownership
        future_map : legion_future_map_t
            A future map that is currently storing the data for
            this argument map
        """
        if handle is not None:
            if future_map is not None:
                raise ValueError(
                    "Cannot supply a 'future_map' with a 'handle'"
                )
            self.handle = handle
        elif future_map is None:
            self.handle = legion.legion_argument_map_create()
        else:
            self.handle = legion.legion_argument_map_from_future_map(
                future_map.handle
            )
        self.points = None

    def __del__(self):
        self.destroy(unordered=True)

    def destroy(self, unordered):
        """
        Eagerly destroy this ArgumentMap before the garbage collector does
        It is illegal to use the ArgumentMap after this call

        Parameters
        ----------
        unordered : bool
            Whether this ArgumentMap is being destroyed outside of the scope
            of the execution of a Legion task or not
        """
        if self.handle is None:
            return
        if unordered:
            _pending_deletions.append((self.handle, type(self)))
        else:
            legion.legion_future_map_destroy(self.handle)
        self.handle = None

    def set_point(self, point, data, size, replace=True):
        """
        Set the point argument in the argument map

        Parameters
        ----------
        point : Point
            The point to set in the argument map
        data : object that implements the Python buffer protocol
            The data to be set for the point
        size : int
            The size of the data in the buffer
        replace : bool
            Whether we should replace this argument if it already exists
        """
        arg = ffi.new("legion_task_argument_t *")
        if data is not None:
            arg[0].args = ffi.from_buffer(data)
            arg[0].arglen = size
        else:
            arg[0].args = ffi.NULL
            arg[0].arglen = 0
        legion.legion_argument_map_set_point(
            self.handle, point.raw(), arg[0], replace
        )
        if self.points is None:
            self.points = list()
        self.points.append(arg)

    def set_future(self, point, future, replace=True):
        """
        Set the point argument in the argument map using a Future

        Parameters
        ----------
        point : Point
            The point to set in the argument map
        future : Future
            The future
        replace : bool
            Whether we should replace this argument if it already exists
        """
        legion.legion_argument_map_set_future(
            self.handle, point.raw(), future.handle, replace
        )


class BufferBuilder(object):
    def __init__(self, type_safe=False):
        """
        A BufferBuilder object is a helpful utility for constructing
        buffers of bytes to pass through to tasks in other languages.
        """

        self.fmt = list()  # struct format string
        self.fmt.append("=")
        self.size = 0
        self.args = list()
        self.string = None
        self.arglen = None
        self.type_safe = type_safe

    def add_arg(self, arg, type_val):
        # Save the type of the object as integer right before it
        # The integer must be matched in the C++ code in the unpack functions
        if self.type_safe:
            self.fmt.append("i")
            self.size += 4
            self.args.append(type_val)
        self.args.append(arg)

    def pack_8bit_int(self, arg):
        self.fmt.append("b")
        self.size += 1
        self.add_arg(arg, legion.LEGION_TYPE_INT8)

    def pack_16bit_int(self, arg):
        self.fmt.append("h")
        self.size += 2
        self.add_arg(arg, legion.LEGION_TYPE_INT16)

    def pack_32bit_int(self, arg):
        self.fmt.append("i")
        self.size += 4
        self.add_arg(arg, legion.LEGION_TYPE_INT32)

    def pack_64bit_int(self, arg):
        self.fmt.append("q")
        self.size += 8
        self.add_arg(arg, legion.LEGION_TYPE_INT64)

    def pack_8bit_uint(self, arg):
        self.fmt.append("B")
        self.size += 1
        self.add_arg(arg, legion.LEGION_TYPE_UINT8)

    def pack_16bit_uint(self, arg):
        self.fmt.append("H")
        self.size += 2
        self.add_arg(arg, legion.LEGION_TYPE_UINT16)

    def pack_32bit_uint(self, arg):
        self.fmt.append("I")
        self.size += 4
        self.add_arg(arg, legion.LEGION_TYPE_UINT32)

    def pack_64bit_uint(self, arg):
        self.fmt.append("Q")
        self.size += 8
        self.add_arg(arg, legion.LEGION_TYPE_UINT64)

    def pack_32bit_float(self, arg):
        self.fmt.append("f")
        self.size += 4
        self.add_arg(arg, legion.LEGION_TYPE_FLOAT32)

    def pack_64bit_float(self, arg):
        self.fmt.append("d")
        self.size += 8
        self.add_arg(arg, legion.LEGION_TYPE_FLOAT64)

    def pack_bool(self, arg):
        self.fmt.append("?")
        self.size += 1
        self.add_arg(arg, legion.LEGION_TYPE_BOOL)

    def pack_16bit_float(self, arg):
        self.fmt.append("h")
        self.size += 2
        self.add_arg(arg, legion.LEGION_TYPE_FLOAT16)

    def pack_char(self, arg):
        self.fmt.append("c")
        self.size += 1
        self.add_arg(bytes(arg.encode("utf-8")), legion.LEGION_TYPE_TOTAL + 1)

    def pack_64bit_complex(self, arg):
        if self.type_safe:
            self.fmt.append("i")
            self.size += 4
            self.args.append(legion.LEGION_TYPE_COMPLEX64)
        self.fmt.append("ff")  # encode complex as two floats
        self.args.append(arg.real)
        self.args.append(arg.imag)

    def pack_128bit_complex(self, arg):
        if self.type_safe:
            self.fmt.append("i")
            self.size += 4
            self.args.append(legion.LEGION_TYPE_COMPLEX128)
        self.fmt.append("dd")  # encode complex as two floats
        self.args.append(arg.real)
        self.args.append(arg.imag)

    def pack_dimension(self, dim):
        self.pack_32bit_int(dim)

    def pack_point(self, point):
        if not isinstance(point, tuple):
            raise ValueError("'point' must be a tuple")
        dim = len(point)
        if dim <= 0:
            raise ValueError("'dim' must be positive")
        if self.type_safe:
            self.pack_32bit_int(dim)
        for p in point:
            self.pack_64bit_int(p)

    def pack_accessor(self, field_id, transform=None, point_transform=None):
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

    def pack_transform(self, transform):
        for x in xrange(0, transform.M):
            for y in xrange(0, transform.N):
                self.pack_64bit_int(transform.trans[x, y])
        for x in xrange(0, transform.M):
            self.pack_64bit_int(transform.offset[x])

    def pack_value(self, value, val_type):
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
        elif np.dtype(val_type) == np.bool:
            self.pack_bool(value)
        elif np.dtype(val_type) == np.float16:
            self.pack_16bit_float(value)
        elif np.dtype(val_type) == np.complex64:
            self.pack_64bit_complex(value)
        elif np.dtype(val_type) == np.complex128:
            self.pack_128bit_complex(value)
        else:
            raise TypeError("Unhandled value type")

    def pack_string(self, string):
        self.pack_32bit_int(len(string))
        for char in string:
            self.pack_char(char)

    def pack_buffer(self, buf):
        self.pack_32bit_uint(buf.get_size())
        self.fmt.append(buf.fmt[1:])
        self.size += buf.size
        self.args.append(*(buf.args))

    # Static member of this class for encoding dtypes
    _dtype_codes = {
        bool: legion.LEGION_TYPE_BOOL,
        np.bool: legion.LEGION_TYPE_BOOL,
        np.bool_: legion.LEGION_TYPE_BOOL,
        np.int8: legion.LEGION_TYPE_INT8,
        np.int16: legion.LEGION_TYPE_INT16,
        np.int: legion.LEGION_TYPE_INT32,
        np.int32: legion.LEGION_TYPE_INT32,
        np.int64: legion.LEGION_TYPE_INT64,
        np.uint8: legion.LEGION_TYPE_UINT8,
        np.uint16: legion.LEGION_TYPE_UINT16,
        np.uint32: legion.LEGION_TYPE_UINT32,
        np.uint64: legion.LEGION_TYPE_UINT64,
        np.float16: legion.LEGION_TYPE_FLOAT16,
        np.float: legion.LEGION_TYPE_FLOAT32,
        np.float32: legion.LEGION_TYPE_FLOAT32,
        np.float64: legion.LEGION_TYPE_FLOAT64,
        np.complex64: legion.LEGION_TYPE_COMPLEX64,
        np.complex128: legion.LEGION_TYPE_COMPLEX128,
    }

    @classmethod
    def encode_dtype(cls, dtype):
        if dtype in cls._dtype_codes:
            return cls._dtype_codes[dtype]
        elif hasattr(dtype, "type") and dtype.type in cls._dtype_codes:
            return cls._dtype_codes[dtype.type]
        raise ValueError(
            str(dtype) + " is not a valid data type for BufferBuilder"
        )

    def pack_dtype(self, dtype):
        self.pack_32bit_int(self.encode_dtype(dtype))

    def get_string(self):
        if self.string is None or self.arglen != len(self.args):
            fmtstr = "".join(self.fmt)
            assert len(fmtstr) == len(self.args) + 1
            self.string = struct.pack(fmtstr, *self.args)
            self.arglen = len(self.args)
        return self.string

    def get_size(self):
        return self.size
