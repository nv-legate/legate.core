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

import platform
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from typing_extensions import Protocol

if TYPE_CHECKING:
    from .store import Store
    from .types import Dtype


class LegateDataInterfaceItem(TypedDict):
    version: int
    data: dict[Field, Array]


class LegateDataInterface(Protocol):
    @property
    def __legate_data_interface__(self) -> LegateDataInterfaceItem:
        ...


class Field:
    def __init__(
        self,
        name: str,
        dtype: Dtype,
        nullable: bool = False,
    ):
        """
        A field is metadata associated with a single array in the legate data
        interface object.

        Parameters
        ----------
        name : str
            Field name
        dtype : Dtype
            The type of the array
        nullable : bool
            Indicates whether the array is nullable
        """

        if nullable:
            raise NotImplementedError("Nullable array is not yet supported")

        self._name = name
        self._dtype = dtype
        self._nullable = nullable

    @property
    def name(self) -> str:
        """
        Returns the array's name

        Returns
        -------
        str
            Name of the field
        """
        return self._name

    @property
    def type(self) -> Dtype:
        """
        Returns the array's data type

        Returns
        -------
        Dtype
            Data type of the field
        """
        return self._dtype

    @property
    def nullable(self) -> bool:
        """
        Indicates whether the array is nullable

        Returns
        -------
        bool
            ``True`` if the array is nullable. ``False`` otherwise.
        """
        return self._nullable


class Array:
    def __init__(
        self,
        dtype: Dtype,
        stores: list[Optional[Store]],
    ) -> None:
        """
        An Array is a collection of one or more Store objects that can
        represent a uniformly typed set of potentially nullable data values.

        Parameters
        ----------
        dtype : Dtype
            The type for the constructed array
        stores : List[Store]
            List of storage objects

        Returns
        -------
        A newly constructed Array
        """
        self._type = dtype
        self._stores = stores

    def stores(self) -> list[Optional[Store]]:
        """
        Return a list of the Store object that represent
        the data stored in this array.
        """
        return self._stores.copy()

    @property
    def type(self) -> Dtype:
        return self._type


class Table(LegateDataInterface):
    def __init__(self, fields: list[Field], columns: list[Array]) -> None:
        """
        A Table is a collection of top-level, equal-length Array
        objects.
        """
        self._fields = fields
        self._columns = columns

    @property
    def __legate_data_interface__(self) -> LegateDataInterfaceItem:
        """
        The Legate data interface allows for different Legate libraries to get
        access to the base Legion primitives that back objects from different
        Legate libraries. It currently requires objects that implement it to
        return a dictionary that contains two members:

        Returns
        -------
        A dictionary with the following entries:

        'version' (required) : int
            An integer showing the version number of this implementation of
            the interface (i.e. 1 for this version)

        'data' (required) : dict[Field, Array]
            An dictionary mapping ``Field`` objects that represent the
            names and types of the field data to ``Array`` objects containing
            Store objects

        """
        result: LegateDataInterfaceItem = {
            "version": 1,
            "data": dict(zip(self._fields, self._columns)),
        }
        return result

    @staticmethod
    def from_arrays(
        names: list[str],
        arrays: list[Array],
    ) -> Table:
        """
        Construct a Table from a list of Arrays.

        Parameters
        ----------
        arrays : List[Array]
            Equal-length arrays that should form the table.
        names : List[str], optional
            Names for the table columns. If not passed, schema must be passed

        Returns
        -------
        Table
        """
        if len(names) != len(arrays):
            raise ValueError(
                f"Length of names ({names}) does not match "
                f"length of arrays ({arrays})"
            )
        fields = [
            Field(name, array.type) for name, array in zip(names, arrays)
        ]
        return Table(fields, arrays.copy())


class Library:
    def __init__(self) -> None:
        """
        This is the abstract class for a Legate library class. It describes
        all the methods that need to be implemented to support a library
        that is registered with the Legate runtime.
        """
        pass

    def get_name(self) -> str:
        """
        Returns a name of the library

        Returns
        -------
        str
            Library name
        """
        raise NotImplementedError("Implement in derived classes")

    def get_shared_library(self) -> Optional[str]:
        """
        Returns the path to the shared library

        Returns
        -------
        str or ``None``
            Path to the shared library
        """
        raise NotImplementedError("Implement in derived classes")

    def get_c_header(self) -> str:
        """
        Returns a compiled C header string for the library

        Returns
        -------
        str
            C header string
        """
        raise NotImplementedError("Implement in derived classes")

    def get_registration_callback(self) -> str:
        """
        Returns the name of a C registration callback for the library

        Returns
        -------
        str
            The name of the C registration callback
        """
        raise NotImplementedError("Implement in derived classes")

    def initialize(self, shared_lib: Any) -> None:
        """
        This is called when this library is added to Legate
        """
        raise NotImplementedError("Implement in derived classes")

    def destroy(self) -> None:
        """
        This is called on shutdown by Legate
        """
        raise NotImplementedError("Implement in derived classes")

    @staticmethod
    def get_library_extension() -> str:
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"
        raise RuntimeError(f"unknown platform {os_name!r}")
