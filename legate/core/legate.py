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

import platform

import pyarrow

from .context import ResourceConfig


class Array(object):
    def __init__(self, dtype, stores, children=None):
        """
        An Array is a collection of one or more Store objects that can
        represent a uniformly typed set of potentially nullable data values.

        Parameters
        ----------
        dtype : pyarrow.DataType
            The type for the constructed array
        stores : List[Store]
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
                "Type's expected number of Store objects "
                "({0}) did not match the passed number "
                "({1}).".format(dtype.num_buffers, len(self._stores))
            )

    def stores(self):
        """
        Return a list of the Store object that represent
        the data stored in this array.
        """
        stores = self._stores.copy()
        for child in self._children:
            stores.extend(child.stores())
        return stores

    @staticmethod
    def from_stores(dtype, stores, children=None):
        """
        Construct an Array from a DataType and a list of Store objects

        Parameters
        ----------
        dtype : DataType
            The type for the constructed array
        stores : List[Store]
            List of storage objects
        children : List[Array]
            Nested type children with length matching type.num_fields

        Returns
        -------
        A newly constructed Array
        """
        return Array(dtype, stores.copy(), children.copy() if children else [])

    @property
    def type(self):
        return self._type

    def __len__(self):
        raise NotImplementedError("Array.__len__")


class Table(object):
    def __init__(self, schema, columns):
        """
        A Table is a collection of top-level, equal-length Array
        objects. It is designed to be as close as possible to the PyArrow
        Table datatype with the only exception being that its data is backed
        by Store object instead of buffers in memory.
        """
        if len(schema.types) != len(columns):
            raise ValueError(
                "Schema expected number of arrays "
                "({0}) did not match the passed number "
                "({1}).".format(len(schema.types), len(columns))
            )
        self._schema = schema
        self._columns = columns

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
        'data' (required) : OrderedDict[Field,Array]
            An ordered dictionary mapping 'Field' objects that represent the
            names and types of the field data to 'Array' objects containing
            Store objects
        """
        result = dict()
        result["version"] = 1
        data = {}
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
        Table : New table with the passed column added.
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
        return Table(pyarrow.schema(fields), columns)

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
        Table : New table with the passed column added.
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
        Table : New table without the columns.
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
        return Table(pyarrow.schema(fields), columns)

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
        Construct a Table from a list of Arrays.

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
        Table
        """
        if schema is None:
            if names is None:
                raise ValueError(
                    "Must pass names or schema when constructing Table"
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
        return Table(schema, arrays.copy())

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
        Table : New table without the column.
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
        return Table(pyarrow.schema(fields), columns)

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
        return Table(pyarrow.schema(fields), self._columns.copy())

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
        Table : New table with the passed column set.
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
        return Table(pyarrow.schema(fields), columns)

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
        return len(self._columns[0])

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


class Library(object):
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

    def get_resource_configuration(self):
        """
        Return a ResourceConfig object that configures the library
        """
        # Return the default configuration
        return ResourceConfig()

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
