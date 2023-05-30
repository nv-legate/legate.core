# Copyright 2023 NVIDIA Corporation
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
""" Control global configuration options with environment variables.

Precedence
~~~~~~~~~~

Setting values are always looked up in the following prescribed order:

immediately supplied values
    These are values that are passed to the setting:

    .. code-block:: python

        settings.consensus(value)

    If ``value`` is not None, then it will be returned, as-is. Otherwise, if
    None is passed, then the setting will continue to look down the search
    order for a value. This is useful for passing optional function paramters
    that are None by default. If the parameter is passed to the function, then
    it will take precedence.

previously user-set values
    If the value is set explicity in code:

    .. code-block:: python

        settings.minified = False

    Then this value will take precedence over other sources. Applications
    may use this ability to set values supplied on the command line, so that
    they take precedence over environment variables.

environment variable
    Values found in the associated environment variables:

    .. code-block:: sh

        LEGATE_CONSENSUS=yes legate script.py

local defaults
    These are default values defined when accessing the setting:

    .. code-block:: python

        settings.concensus(default=True)

    Local defaults have lower precendence than every other setting mechanism
    except global defaults.

global defaults
    These are default values defined by the setting declarations. They have
    lower precedence than every other setting mechanism.

If no value is obtained after searching all of these locations, then a
RuntimeError will be raised.

"""
from __future__ import annotations

import os
from typing import Any, Generic, Type, TypeVar, Union

from typing_extensions import TypeAlias

__all__ = (
    "convert_str",
    "convert_bool",
    "convert_str_seq",
    "PrioritizedSetting",
    "Settings",
)


class _Unset:
    pass


T = TypeVar("T")


Unset: TypeAlias = Union[T, Type[_Unset]]


def convert_str(value: str) -> str:
    """Return a string as-is."""
    return value


def convert_int(value: str) -> int:
    """Return an integer value"""
    return int(value)


def convert_bool(value: bool | str) -> bool:
    """Convert a string to True or False.

    If a boolean is passed in, it is returned as-is. Otherwise the function
    maps the strings "0" -> False and "1" -> True.

    Args:
        value (str):
            A string value to convert to bool

    Returns:
        bool

    Raises:
        ValueError

    """
    if isinstance(value, bool):
        return value

    val = value.lower()
    if val == "1":
        return True
    if val == "0":
        return False

    raise ValueError(f'Cannot convert {value!r} to bool, use "0" or "1"')


def convert_str_seq(
    value: list[str] | tuple[str, ...] | str
) -> tuple[str, ...]:
    """Convert a string to a list of strings.

    If a list or tuple is passed in, it is returned as-is.

    Args:
        value (seq[str] or str) :
            A string to convert to a list of strings

    Returns
        list[str]

    Raises:
        ValueError

    """
    if isinstance(value, (list, tuple)):
        return tuple(value)

    try:
        return tuple(value.split(","))
    except Exception:
        raise ValueError(f"Cannot convert {value} to list value")


class SettingBase:
    def __init__(
        self,
        name: str,
        default: Unset[T] = _Unset,
        convert: Any | None = None,
        help: str = "",
    ) -> None:
        self._default = default
        self._convert = convert if convert else convert_str
        self._help = help
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def help(self) -> str:
        return self._help

    @property
    def default(self) -> Unset[T]:
        return self._default

    @property
    def convert_type(self) -> str:
        if self._convert is convert_str:
            return "str"
        if self._convert is convert_int:
            return "int"
        if self._convert is convert_bool:
            return 'bool ("0" or "1")'
        if self._convert is convert_str_seq:
            return "tuple[str, ...]"
        raise RuntimeError("unreachable")


class PrioritizedSetting(Generic[T], SettingBase):
    """Return a value for a global setting according to configuration
    precedence.

    The following methods are searched in order for the setting:

    4. immediately supplied values
    3. previously user-set values (e.g. set from command line)
    2. environment variable
    1. local defaults
    0. global defaults

    If a value cannot be determined, a RuntimeError is raised.

    The ``env_var`` argument specifies the name of an environment to check for
    setting values, e.g. ``"LEGATE_CHECK_CYCLE"``.

    The optional ``default`` argument specified an implicit default value for
    the setting that is returned if no other methods provide a value.

    A ``convert`` agument may be provided to convert values before they are
    returned.
    """

    _user_value: Unset[str | T]

    def __init__(
        self,
        name: str,
        env_var: str | None = None,
        default: Unset[T] = _Unset,
        convert: Any | None = None,
        help: str = "",
    ) -> None:
        super().__init__(name, default, convert, help)
        self._env_var = env_var
        self._user_value = _Unset

    def __call__(
        self, value: T | str | None = None, default: Unset[T] = _Unset
    ) -> T:
        """Return the setting value according to the standard precedence.

        Args:
            value (any, optional):
                An optional immediate value. If not None, the value will
                be converted, then returned.

            default (any, optional):
                An optional default value that only takes precendence over
                implicit default values specified on the property itself.

        Returns:
            str or int or float

        Raises:
            RuntimeError
        """

        # 4. immediate values
        if value is not None:
            return self._convert(value)

        # 3. previously user-set value
        if self._user_value is not _Unset:
            return self._convert(self._user_value)

        # 2. environment variable
        if self._env_var and self._env_var in os.environ:
            return self._convert(os.environ[self._env_var])

        # 1. local defaults
        if default is not _Unset:
            return self._convert(default)

        # 0. global defaults
        if self._default is not _Unset:
            return self._convert(self._default)

        raise RuntimeError(
            f"No configured value found for setting {self._name!r}"
        )

    def __get__(
        self, instance: Any, owner: type[Any]
    ) -> PrioritizedSetting[T]:
        return self

    def __set__(self, instance: Any, value: str | T) -> None:
        self.set_value(value)

    def set_value(self, value: str | T) -> None:
        """Specify a value for this setting programmatically.

        A value set this way takes precedence over all other methods except
        immediate values.

        Args:
            value (str or int or float):
                A user-set value for this setting

        Returns:
            None
        """
        # It is usually not advised to store any data directly on descriptors,
        # since they are shared by all instances. But in our case we only ever
        # have a single instance of a given settings object.
        self._user_value = value

    def unset_value(self) -> None:
        """Unset the previous user value such that the priority is reset."""
        self._user_value = _Unset

    @property
    def env_var(self) -> str | None:
        return self._env_var


class EnvOnlySetting(Generic[T], SettingBase):
    """Return a value for a global environment variable setting.

    A ``convert`` agument may be provided to convert values before they are
    returned.
    """

    def __init__(
        self,
        name: str,
        env_var: str,
        default: Unset[T] = _Unset,
        test_default: Unset[T] = _Unset,
        convert: Any | None = None,
        help: str = "",
    ) -> None:
        super().__init__(name, default, convert, help)
        self._test_default = test_default
        self._env_var = env_var

    def __call__(self) -> T:
        if self._env_var in os.environ:
            return self._convert(os.environ[self._env_var])

        # unfortunate
        test = convert_bool(os.environ.get("LEGATE_TEST", False))

        if test and self.test_default is not _Unset:
            return self._convert(self.test_default)

        return self._convert(self.default)

    def __get__(self, instance: Any, owner: type[Any]) -> EnvOnlySetting[T]:
        return self

    @property
    def env_var(self) -> str | None:
        return self._env_var

    @property
    def test_default(self) -> Unset[T]:
        return self._test_default


class Settings:
    pass
