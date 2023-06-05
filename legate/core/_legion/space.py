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

from typing import TYPE_CHECKING, Any, Optional, Union

from .. import ffi, legion
from ..types import Dtype
from .env import LEGATE_MAX_FIELDS
from .field import FieldID
from .future import Future
from .geometry import Domain
from .pending import _pending_unordered

if TYPE_CHECKING:
    from . import IndexPartition, Rect


class IndexSpace:
    _logical_handle: Any

    def __init__(
        self,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        handle: Any,
        parent: Optional[IndexPartition] = None,
        owned: bool = True,
    ) -> None:
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
        self.children: Union[set[IndexPartition], None] = None
        self.owned = owned
        self._domain: Optional[Domain] = None
        if owned and self.parent is not None and not self.parent.owned:
            raise ValueError(
                "IndexSpace can only own its handle if the parent "
                "IndexPartition also owns its handle"
            )

    def __del__(self) -> None:
        # We only need to delete top-level index spaces
        # Ignore any deletions though that occur after the task is done
        if self.owned and self.parent is None:
            self.destroy(unordered=True)

    def _can_delete(self) -> bool:
        if not self.owned:
            return False
        if self.parent is not None:
            return self.parent.parent._can_delete()
        # Must be owned at the root to enable deletion
        return self.owned

    def add_child(self, child: IndexPartition) -> None:
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

    def destroy(self, unordered: bool = False) -> None:
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
                    (
                        (self.handle, getattr(self, "_logical_handle")),
                        type(self),
                    )
                )
            else:
                _pending_unordered[self.context].append(
                    ((self.handle, None), type(self))
                )
        else:
            legion.legion_index_space_destroy_unordered(
                self.runtime, self.context, self.handle, False
            )

    def get_root(self) -> IndexSpace:
        """
        Find the root of IndexSpace tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    @property
    def domain(self) -> Domain:
        """
        Return a Domain that represents the points in this index space
        """
        if self._domain is None:
            self._domain = Domain(
                legion.legion_index_space_get_domain(self.runtime, self.handle)
            )
        return self._domain

    def get_bounds(self) -> Rect:
        """
        Return a Rect that represents the upper bounds of the IndexSpace.
        """
        return self.domain.rect

    def get_volume(self) -> float:
        """
        Return the total number of points in the IndexSpace
        """
        return self.domain.get_volume()

    def get_dim(self) -> int:
        """
        Return the dimension of the IndexSpace
        """
        if self._domain is not None:
            return self._domain.dim
        return legion.legion_index_space_get_dim(self.handle)


class FieldSpace:
    def __init__(
        self,
        context: legion.legion_context_t,
        runtime: legion.legion_runtime_t,
        handle: Optional[Any] = None,
        owned: bool = True,
    ) -> None:
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
        self.fields: dict[int, Any] = dict()
        self.owned = owned

    def __del__(self) -> None:
        # Only delete this if the task is still executing otherwise we leak it
        if self.owned:
            self.destroy(unordered=True)

    @property
    def has_space(self) -> bool:
        return len(self.fields) < LEGATE_MAX_FIELDS

    def allocate_field(
        self,
        size_or_type: Any,
        field_id: int = legion.legion_auto_generate_id(),
    ) -> int:
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
        if isinstance(size_or_type, Dtype):
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
        self, ctype: Any, field_id: int = legion.legion_auto_generate_id()
    ) -> int:
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
        self, dtype: Any, field_id: int = legion.legion_auto_generate_id()
    ) -> int:
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
        self, future: Future, field_id: int = legion.legion_auto_generate_id()
    ) -> int:
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

    def destroy_field(self, field_id: int, unordered: bool = False) -> None:
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

    def get_type(self, field_id: int) -> Any:
        """
        Return the type of the object used to create the field.
        """
        return self.fields[field_id]

    def __len__(self) -> int:
        return len(self.fields)

    def destroy(self, unordered: bool = False) -> None:
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
