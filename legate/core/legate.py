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

from __future__ import absolute_import, division, print_function

import collections
import gc
import inspect
import os
import platform
import struct
import sys

import pyarrow

from legion_cffi import ffi  # Make sure we only have one ffi instance
from legion_cffi import lib as legion
from legion_top import cleanup_items, top_level

from legate.core.legion import (
    BufferBuilder,
    FieldID,
    Future,
    IndexTask,
    Rect,
    Region,
    legate_task_postamble,
    legate_task_preamble,
)


# Helper methods for python 3 support
def _itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()


def get_script_dir(follow_symlinks=True):
    if getattr(sys, "frozen", False):  # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)


class LegateStore(object):
    def __init__(self):
        """
        Unlike in Arrow where all data is backed by objects that
        implement the Python Buffer protocol, in Legate data is backed
        by objects that support the LegateStore interface. The LegateStore
        interface allows clients to access the underlying primitives that
        represent the data for a LegateArray object.
        """
        pass

    @property
    def type(self):
        """
        Return the type of the data in this storage primitive
        """

    @property
    def kind(self):
        """
        Return the type of the Legion storage object backing the
        data in this storage object: either Future, FutureMap,
        (Region,FieldID), or (Region,int)
        """
        raise NotImplementedError("implement in derived classes")

    @property
    def storage(self):
        """
        Return the Legion storage objects actually backing the
        data for this LegateStore. These will have exactly the
        type specified in by 'kind'
        """
        raise NotImplementedError("implement in derived classes")


class LegateArray(object):
    def __init__(self, dtype, stores, children=None):
        """
        An Array is a collection of one or more LegateStore objects that can
        represent a uniformly typed set of potentially nullable data values.

        Construct an Array from a DataType and a list of LegateStore objects

        Parameters
        ----------
        dtype : DataType
            The type for the constructed array
        stores : List[LegateStore]
            List of storage objects
        children : List[Array]
            Nested type children with length matching type.num_fields

        Returns
        -------
        A newly constructed Array
        """
        self._type = dtype
        self._stores = stores
        self._children = children or []

        if dtype.num_fields != len(self._children):
            raise ValueError(
                "Type's expected number of children "
                "({0}) did not match the passed number "
                "({1}).".format(dtype.num_fields, len(self._children))
            )

        if dtype.num_buffers != len(self._stores):
            raise ValueError(
                "Type's expected number of LegateStore objects "
                "({0}) did not match the passed number "
                "({1}).".format(dtype.num_buffers, len(self._stores))
            )

        # Check that all the stores are the same type and logical region
        all_stores = self.stores()
        all_store_types = set(type(store) for store in all_stores) - {
            type(None)
        }
        if len(all_store_types) > 1:
            raise TypeError("All stores of an Array must be the same type")

        # By convention, we have a valid store at 1.
        if stores[1].kind in ((Region, FieldID), (Region, int)):
            self._region = stores[1].storage[0]
        else:
            self._region = None

    def stores(self):
        """
        Return a list of the LegateStore object that represent
        the data stored in this array.
        """
        stores = self._stores.copy()
        for child in self._children:
            stores.extend(child.stores())
        return stores

    @staticmethod
    def from_stores(dtype, stores, children=None):
        """
        Construct an Array from a DataType and a list of LegateStore objects

        Parameters
        ----------
        dtype : DataType
            The type for the constructed array
        stores : List[LegateStore]
            List of storage objects
        children : List[Array]
            Nested type children with length matching type.num_fields

        Returns
        -------
        A newly constructed Array
        """
        return LegateArray(
            dtype, stores.copy(), children.copy() if children else []
        )

    @property
    def type(self):
        return self._type

    @property
    def region(self):
        return self._region


class LegateTable(object):
    def __init__(self, schema, columns):
        """
        A LegateTable is collection of top-level, equal-length LegateStore
        objects. It is designed to be as close as possible to the PyArrow
        Table datatype with the only exception being that its data is backed
        by LegateStore object instead of buffers in memory.
        """
        if len(schema.types) != len(columns):
            raise ValueError(
                "Schema expected number of arrays "
                "({0}) did not match the passed number "
                "({1}).".format(len(schema.types), len(columns))
            )
        self._schema = schema
        self._columns = columns
        region = self._columns[0].region
        for column in self._columns:
            if column.region is not region:
                raise ValueError(
                    "All Arrays in a LegateTable must "
                    "have the same logical region"
                )

    @property
    def __legate_data_interface__(self):
        """
        The Legate data interface allows for different Legate libraries to get
        access to the base Legion primitives that back objects from different
        Legate libraries. It currently requires objects that implement it to
        return a dictionary that contains two integer members:

        Returns
        -------
        A dictionary with the following entries:
        'version' (required) : int
            An integer showing the version number of this implementation of
            the interface (i.e. 1 for this version)
        'data' (required) : OrderedDict[Field,LegateArray]
            An ordered dictionary mapping 'Field' objects that represent the
            names and types of the field data to 'Array' objects containing
            LegateStore objects
        """
        result = dict()
        result["version"] = 1
        data = collections.OrderedDict()
        for index, column in enumerate(self._columns):
            data[self._schema.field(index)] = column
        result["data"] = data
        return result

    def add_column(self, index, field, column):
        """
        Add column to Table at position.
        A new table is returned with the column added, the original table
        object is left unchanged.

        Parameters
        ----------
        index : int
            Index to place the column at.
        field : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array

        Returns
        -------
        LegateTable : New table with the passed column added.
        """
        if index < 0 or index > len(self._columns):
            raise ValueError(
                "index " + str(index) + " out of bounds in 'add_column'"
            )
        if not isinstance(field, pyarrow.Field):
            field = pyarrow.field(field, column.type)
        fields = []
        columns = []
        for idx, col in enumerate(self._columns):
            if idx == index:
                fields.append(field)
                columns.append(column)
            fields.append(self._schema.field(index))
            columns.append(col)
        if index == len(self._columns):
            fields.append(field)
            columns.append(column)
        return LegateTable(pyarrow.schema(fields), columns)

    def append_column(self, field, column):
        """
        Append column at end of columns.

        Parameters
        ----------
        field : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array

        Returns
        -------
        LegateTable : New table with the passed column added.
        """
        return self.add_column(len(self._columns), field, column)

    def _ensure_integer_index(self, i):
        """
        Ensure integer index (convert string column name to integer if needed).
        """
        if isinstance(i, (bytes, str)):
            field_indices = self._schema.get_all_field_indices(i)

            if len(field_indices) == 0:
                raise KeyError(
                    'Field "{}" does not exist in table schema'.format(i)
                )
            elif len(field_indices) > 1:
                raise KeyError(
                    'Field "{}" exists {} times in table schema'.format(
                        i, len(field_indices)
                    )
                )
            else:
                return field_indices[0]
        elif isinstance(i, int):
            return i
        else:
            raise TypeError("Index must either be string or integer")

    def column(self, index):
        """
        Select a column by its column name, or numeric index.

        Parameters
        ----------
        i : int or string
            The index or name of the column to retrieve.

        Returns
        -------
        Array
        """
        index = self._ensure_integer_index(index)
        if index < 0 or index >= len(self._columns):
            raise ValueError(
                "index " + str(index) + " out of bounds in 'add_column'"
            )
        return self._columns[index]

    def drop(self, columns):
        """
        Drop one or more columns and return a new table.

        Parameters
        ----------
        columns : list of str
            List of field names referencing existing columns.

        Raises
        ------
        KeyError : if any of the passed columns name are not existing.

        Returns
        -------
        LegateTable : New table without the columns.
        """
        indices = []
        for col in columns:
            idx = self._schema.get_field_index(col)
            if idx == -1:
                raise KeyError("Column {!r} not found".format(col))
            indices.append(idx)

        indices.sort()
        fields = []
        columns = []
        skip_index = 0
        for idx, col in enumerate(self._columns):
            if skip_index < len(indices) and idx == indices[skip_index]:
                continue
            fields.append(self._schema.field(idx))
            columns.append(col)
        return LegateTable(pyarrow.schema(fields), columns)

    def field(self, index):
        """
        Select a schema field by its column name or numeric index.

        Parameters
        ----------
        index : int or string
            The index or name of the field to retrieve.

        Returns
        -------
        pyarrow.Field
        """
        return self._schema.field(index)

    @staticmethod
    def from_arrays(arrays, names=None, schema=None, metadata=None):
        """
        Construct a LegateTable from a list of Legate Arrays.

        Parameters
        ----------
        arrays : List[Array]
            Equal-length arrays that should form the table.
        names : List[str], optional
            Names for the table columns. If not passed, schema must be passed
        schema : Schema, default None
            Schema for the created table. If not passed, names must be passed
        metadata : dict or Mapping, default None
            Optional metadata for the schema (if inferred).

        Returns
        -------
        LegateTable
        """
        if schema is None:
            if names is None:
                raise ValueError(
                    "Must pass names or schema when constructing LegateTable"
                )
            if len(names) != len(arrays):
                raise ValueError(
                    "Length of names ({}) does not match "
                    "length of arrays ({})".format(len(names), len(arrays))
                )
            fields = [
                pyarrow.field(names[index], array.type)
                for index, array in enumerate(arrays)
            ]
            schema = pyarrow.schema(fields, metadata)
        else:
            if names is not None:
                raise ValueError("Cannot pass both schema and names")
            if metadata is not None:
                raise ValueError("Cannot pass both schema and metadata")
            if len(schema) != len(arrays):
                raise ValueError("Schema and number of arrays unequal")
            for index, array in enumerate(arrays):
                if not schema[index].type.equals(array.type):
                    raise TypeError("Schema type and Array type must match")
        return LegateTable(schema, arrays.copy())

    def itercolumns(self):
        """
        Iterator over all columns in their numerical order.

        Yields
        ------
        Array
        """
        for column in self._columns:
            yield column

    def remove_column(self, index):
        """
        Create new Table with the indicated column removed.

        Parameters
        ----------
        index : int
            Index of column to remove.

        Returns
        -------
        LegateTable : New table without the column.
        """
        if index < 0 or index >= len(self._columns):
            raise ValueError(
                "index " + str(index) + " out of bounds in 'remove_column'"
            )
        fields = []
        columns = []
        for idx, col in enumerate(self._columns):
            if idx == index:
                continue
            fields.append(self._schema.field(idx))
            columns.append(col)
        return LegateTable(pyarrow.schema(fields), columns)

    def rename_columns(self, names):
        """
        Create new table with columns renamed to provided names.
        """
        if len(names) != len(self._columns):
            raise ValueError(
                "Number of names does not match number of columns"
            )
        fields = []
        for index in range(len(self._schema)):
            field = self._schema.field(index)
            fields.append(field.with_name(names[index]))
        return LegateTable(pyarrow.schema(fields), self._columns.copy())

    def set_column(self, index, field, column):
        """
        Replace column in Table at position.

        Parameters
        ----------
        index : int
            Index to place the column at.
        field : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array

        Returns
        -------
        Legate Table : New table with the passed column set.
        """
        if index < 0 or index >= len(self._columns):
            raise ValueError(
                "index " + str(index) + " out of bounds in 'set_column'"
            )
        if not isinstance(field, pyarrow.Field):
            field = pyarrow.field(field, column.type)
        fields = []
        columns = []
        for idx, col in enumerate(self._columns):
            if idx == index:
                fields.append(field)
                columns.append(column)
            else:
                fields.append(self._schema.field(idx))
                columns.append(column)
        return LegateTable(pyarrow.schema(fields), columns)

    @property
    def column_names(self):
        """
        Return a list of the names of the table's columns.
        """
        return [
            self._schema.field(index).name
            for index in range(len(self._schema))
        ]

    @property
    def columns(self):
        """
        Return a list of the columns in numerical order.
        """
        return self._columns.copy()

    @property
    def num_columns(self):
        """
        Return the number of columns in the table.
        """
        return len(self._columns)

    @property
    def num_rows(self):
        """
        Return the number of rows in the table.
        """
        if len(self._columns) == 0:
            return 0
        stores = self._columns[0][1].stores()
        assert len(stores) > 0
        if stores[0].is_future:
            return 1
        return stores[0].region.index_space.get_volume()

    @property
    def schema(self):
        """
        Return the schema of the table and its columns.
        """
        return self._schema

    @property
    def shape(self):
        """
        Dimensions of the table: (#rows, #columns).

        Returns
        -------
        (int, int) Number of rows and number of columns.
        """
        return (self.num_rows, self.num_columns)


class LegateLibrary(object):
    def __init__(self):
        """
        This is the abstract class for a Legate library class. It describes
        all the methods that need to be implemented to support a library
        that is registered with the Legate runtime.
        """
        pass

    def get_name(self):
        """
        Return a string name describing this library
        """
        raise NotImplementedError("Implement in derived classes")

    def get_shared_library(self):
        """
        Return the name of the shared library
        """
        raise NotImplementedError("Implement in derived classes")

    def get_c_header(self):
        """
        Return a compiled C string header for this library
        """
        raise NotImplementedError("Implement in derived classes")

    def get_registration_callback(self):
        """
        Return the name of a C registration callback for this library
        """
        raise NotImplementedError("Implement in derived classes")

    def initialize(self, shared_lib=None):
        """
        This is called when this library is added to Legate
        """
        raise NotImplementedError("Implement in derived classes")

    def destroy(self):
        """
        This is called on shutdown by Legate
        """
        raise NotImplementedError("Implement in derived classes")

    @staticmethod
    def get_library_extension():
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"


class Attachment(object):
    def __init__(self, ptr, extent, region, field_id):
        self.ptr = ptr
        self.extent = extent
        self.end = ptr + extent - 1
        self.region = region
        self.field_id = field_id
        self.count = 1

    def overlaps(self, start, stop):
        if self.end < start:
            return False
        if stop < self.ptr:
            return False
        return True

    def equals(self, ptr, extent):
        return ptr == self.ptr and extent == self.extent


class LegateCore(LegateLibrary):
    def __init__(self):
        self._legate_dir = get_script_dir()
        self._libraries = list()
        # TODO: provide support for an interval tree here for faster lookups
        self._attachments = None
        self._runtime = None
        self._cuda_libraries = None
        self._task_offset = None
        self._mapper_id = None
        self._num_gpus = None
        # Load our shared object
        self.load_library(self)
        # Record whether we need to run finalize tasks
        # Key off whether we are being loaded in a context or not
        try:
            # Do this first to detect if we're not in the top-level task
            self._context = top_level.context[0]
            runtime = legion.legion_runtime_get_runtime()
            legate_task_preamble(runtime, self._context)
            self._finalize_tasks = True
        except AttributeError:
            self._finalize_tasks = False
            self._context = None

    def get_name(self):
        return "legate.core"

    def get_shared_library(self):
        from legate.core.install_info import libpath

        libname = "liblgcore" + self.get_library_extension()
        return os.path.join(libpath, libname)

    def get_c_header(self):
        from legate.core.install_info import header

        return header

    def get_registration_callback(self):
        return "legate_core_perform_registration"

    def get_runtime(self):
        # Need this to handle the case where we are not control replicated
        if self._runtime is None:
            self._runtime = legion.legion_runtime_get_runtime()
        return self._runtime

    def get_context(self):
        return self._context

    def get_mapper_id(self):
        if self._mapper_id is None:
            self._mapper_id = legion.legion_runtime_generate_library_mapper_ids(  # noqa: E501
                self.get_runtime(), self.get_name().encode("utf-8"), 1
            )
        return self._mapper_id

    def get_task_id(self, task_id):
        if self._task_offset is None:
            self._task_offset = (
                legion.legion_runtime_generate_library_task_ids(  # noqa: E501
                    self.get_runtime(),
                    self.get_name().encode("utf-8"),
                    self._lib.LEGATE_CORE_NUM_TASK_IDS,
                )
            )
        return self._task_offset + task_id

    def get_num_gpus(self):
        if self._num_gpus is None:
            mapper_id = self.get_mapper_id()
            future = Future(
                legion.legion_runtime_select_tunable_value(
                    self.get_runtime(),
                    self.get_context(),
                    self._lib.LEGATE_CORE_TUNABLE_TOTAL_GPUS,
                    mapper_id,
                    0,
                )
            )
            self._num_gpus = struct.unpack_from("i", future.get_buffer())[0]
        return self._num_gpus

    def initialize(self, shared_lib):
        self._lib = shared_lib
        shared_lib.legate_parse_config()

    def destroy(self):
        # Destroy all the client libraries first
        for library in self._libraries:
            library.destroy()
        # Clean up our attachments so that they can be collected
        self._attachments = None
        self._lib.legate_shutdown()
        if self._finalize_tasks:
            num_gpus = self.get_num_gpus()
            if num_gpus > 0:
                # Launch a finalization task to clean up cuda libraries
                domain = Rect((num_gpus,))
                task = IndexTask(
                    self.get_task_id(self._lib.LEGATE_CORE_FINALIZE_TASK_ID),
                    domain,
                    mapper=self.get_mapper_id(),
                    tag=self._lib.LEGATE_GPU_VARIANT,
                )
                task.launch(self.get_runtime(), self.get_context())
            # Run a gc and then end the legate task
            gc.collect()
            legate_task_postamble(self.get_runtime(), self.get_context())

    def add_library(self, library):
        self.load_library(library)
        self._libraries.append(library)

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

    def add_attachment(self, ptr, extent, region, field_id):
        key = (ptr, extent)
        if self._attachments is None:
            self._attachments = dict()
        elif key in self._attachments:
            # If we find exactly the match then we know by definition that
            # nobody overlaps with this attachment or it wouldn't exist
            self._attachments[key].count += 1
            return
        # Otherwise iterate over attachments and look for aliases which are bad
        end = ptr + extent - 1
        for attachment in _itervalues(self._attachments):
            if attachment.overlaps(ptr, end):
                assert not attachment.equals(ptr, extent)
                raise RuntimeError(
                    "Illegal aliased attachments not supported by " "Legate"
                )
        self._attachments[key] = Attachment(ptr, extent, region, field_id)

    def find_attachment(self, ptr, extent):
        if self._attachments is None:
            return None
        key = (ptr, extent)
        if key in self._attachments:
            attachment = self._attachments[key]
            assert attachment.count > 0
            return (attachment.region, attachment.field_id)
        # Otherwise look for aliases which are bad
        end = ptr + extent - 1
        for attachment in _itervalues(self._attachments):
            if attachment.overlaps(ptr, end):
                assert not attachment.equals(ptr, extent)
                raise RuntimeError(
                    "Illegal aliased attachments not supported by " "Legate"
                )
        return None

    def remove_attachment(self, ptr, extent):
        key = (ptr, extent)
        if key not in self._attachments:
            raise RuntimeError("Unable to find attachment to remove")
        attachment = self._attachments[key]
        assert attachment.count > 0
        if attachment.count == 1:
            del self._attachments[key]
        else:
            attachment.count -= 1

    def initialize_cuda_library(self, libname, block):
        if not isinstance(libname, str):
            raise TypeError("CUDA library name must be a string")
        # If we don't have a context then we'll ignore this request
        # because someone else will be doing the loading
        try:
            top_level.context
        except AttributeError:
            return
        # Figure out how many GPUs we have to load this library on
        num_gpus = self.get_num_gpus()
        # There's nothing to do if there are no GPUs
        if num_gpus == 0:
            return
        if self._cuda_libraries is None:
            self._cuda_libraries = dict()
        libname = libname.lower()
        # See if it is already loaded
        if libname in self._cuda_libraries:
            # If it's been loaded see if we need to wait for it
            if block:
                future_map = self._cuda_libraries[libname]
                if future_map is not None:
                    future_map.wait()
                    # Clear it so no one waits afterwards
                    self._cuda_libraries[libname] = None
            return
        argbuf = BufferBuilder()
        if libname == "cublas":
            argbuf.pack_32bit_int(self._lib.LEGATE_CORE_RESOURCE_CUBLAS)
        elif libname == "cudnn":
            argbuf.pack_32bit_int(self._lib.LEGATE_CORE_RESOURCE_CUDNN)
        elif libname == "cudf":
            argbuf.pack_32bit_int(self._lib.LEGATE_CORE_RESOURCE_CUDF)
        elif libname == "cuml":
            argbuf.pack_32bit_int(self._lib.LEGATE_CORE_RESOURCE_CUML)
        else:
            raise ValueError("Unsupported cuda library" + libname)
        # Launch an initialization task to load the library
        domain = Rect((num_gpus,))
        task = IndexTask(
            self.get_task_id(self._lib.LEGATE_CORE_INITIALIZE_TASK_ID),
            domain,
            data=argbuf.get_string(),
            size=argbuf.get_size(),
            mapper=self.get_mapper_id(),
            tag=self._lib.LEGATE_GPU_VARIANT,
        )
        future_map = task.launch(self.get_runtime(), self.get_context())
        self._cuda_libraries[libname] = future_map
        if block:
            future_map.wait()
            self._cuda_libraries[libname] = None


_core = LegateCore()


def _cleanup_legate():
    global _core
    _core.destroy()
    del _core
    gc.collect()


cleanup_items.append(_cleanup_legate)


def get_legion_runtime():
    return _core.get_runtime()


def get_legion_context():
    try:
        return top_level.context[0]
    except AttributeError:
        return None


def legate_add_library(library):
    _core.add_library(library)


# These functions provide support for deduplicating attaches across libraries
def legate_add_attachment(ptr, extent, region, field_id):
    _core.add_attachment(ptr, extent, region, field_id)


def legate_find_attachment(ptr, extent):
    return _core.find_attachment(ptr, extent)


def legate_remove_attachment(ptr, extent):
    _core.remove_attachment(ptr, extent)


def legate_initialize_cuda_library(libname, block=True):
    _core.initialize_cuda_library(libname, block)


def legate_root_dir():
    return _core._legate_dir
