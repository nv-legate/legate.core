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

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from typing import Any, Iterable, Literal, Sequence, Type, TypeVar, Union

from typing_extensions import TypeAlias

LEGION_WARNING = """

All Legate programs must be run with a legion_python interperter. We
recommend that you use the Legate driver script "bin/legate" found
in the installation directory to launch Legate programs as it
provides easy-to-use flags for invoking legion_python. You can see
options for using the driver script with "bin/legate --help". You
can also invoke legion_python directly.

Use "bin/legate --verbose ..." to see some examples of how to call
legion_python directly.
"""


def has_legion_context() -> bool:
    """Determine whether we are running in legion_python.

    Returns
        bool : True if running in legion_python, otherwise False

    """
    try:
        from legion_cffi import lib

        return lib.legion_runtime_has_context()
    except (ModuleNotFoundError, AttributeError):
        return False


def check_legion(msg: str = LEGION_WARNING) -> None:
    """Raise an error if we are not running in legion_python."""
    if not has_legion_context():
        raise RuntimeError(msg)


class _UnsetType:
    def __repr__(self) -> str:
        return "Unset"


Unset = _UnsetType()

_T = TypeVar("_T")
NotRequired = Union[_UnsetType, _T]


def entries(obj: Any) -> Iterable[tuple[str, Any]]:
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is not Unset:
            yield (f.name, value)


# https://docs.python.org/3/library/argparse.html#action
ActionType: TypeAlias = Literal[
    "store",
    "store_const",
    "store_true",
    "append",
    "append_const",
    "count",
    "help",
    "version",
    "extend",
]

# https://docs.python.org/3/library/argparse.html#nargs
NargsType: TypeAlias = Literal["?", "*", "+", "..."]


@dataclass(frozen=True)
class ArgSpec:
    action: NotRequired[ActionType] = Unset
    nargs: NotRequired[Union[int, NargsType]] = Unset
    const: NotRequired[Any] = Unset
    default: NotRequired[Any] = Unset
    type: NotRequired[Type[Any]] = Unset
    choices: NotRequired[Sequence[Any]] = Unset
    help: NotRequired[str] = Unset
    metavar: NotRequired[str] = Unset
    dest: str = Unset


@dataclass(frozen=True)
class Argument:
    name: str
    spec: ArgSpec


def parse_command_args(libname: str, args: Iterable[Argument]) -> Namespace:
    """ """
    if not libname.isidentifier():
        raise ValueError(
            f"Invalid library {libname!r} for command line arguments"
        )

    parser = ArgumentParser(prog=libname, allow_abbrev=False)

    for arg in args:
        argname = f"-{libname}:{arg.name}"
        kwargs = dict(entries(arg.spec))
        parser.add_argument(argname, **kwargs)

    args, extra = parser.parse_known_args()

    sys.argv = sys.argv[:1] + extra

    return args
