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
import warnings
from argparse import Action, ArgumentParser, Namespace
from dataclasses import dataclass, fields
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import TypeAlias


class _UnsetType:
    pass


Unset = _UnsetType()


T = TypeVar("T")

NotRequired = Union[_UnsetType, T]


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
    dest: str
    action: NotRequired[ActionType] = "store_true"
    nargs: NotRequired[Union[int, NargsType]] = Unset
    const: NotRequired[Any] = Unset
    default: NotRequired[Any] = Unset
    type: NotRequired[Type[Any]] = Unset
    choices: NotRequired[Sequence[Any]] = Unset
    help: NotRequired[str] = Unset
    metavar: NotRequired[str] = Unset


@dataclass(frozen=True)
class Argument:
    name: str
    spec: ArgSpec


def entries(obj: Any) -> Iterable[tuple[str, Any]]:
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is not Unset:
            yield (f.name, value)


class MultipleChoices(Generic[T]):
    """A container that reports True for any item or subset inclusion.

    Parameters
    ----------
    choices: Iterable[T]
        The values to populate the containter.

    Examples
    --------

    >>> choices = MultipleChoices(["a", "b", "c"])

    >>> "a" in choices
    True

    >>> ("b", "c") in choices
    True

    """

    def __init__(self, choices: Iterable[T]) -> None:
        self._choices = set(choices)

    def __contains__(self, x: Union[T, Sequence[T]]) -> bool:
        if isinstance(x, (list, tuple)):
            return set(x).issubset(self._choices)
        return x in self._choices

    def __iter__(self) -> Iterator[T]:
        return self._choices.__iter__()


class ExtendAction(Action, Generic[T]):
    """A custom argparse action to collect multiple values into a list."""

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence[T], None],
        option_string: Union[str, None] = None,
    ) -> None:
        items = getattr(namespace, self.dest) or []
        if isinstance(values, (list, tuple)):
            items.extend(values)
        else:
            items.append(values)
        # removing any duplicates before storing
        setattr(namespace, self.dest, list(set(items)))


def parse_library_command_args(
    libname: str, args: Iterable[Argument]
) -> Namespace:
    """ """
    if not libname.isidentifier():
        raise ValueError(
            f"Invalid library {libname!r} for command line arguments"
        )

    parser = ArgumentParser(
        prog=f"<{libname} program>", add_help=False, allow_abbrev=False
    )

    lib_prefix = f"-{libname}:"

    argnames = [arg.name for arg in args]

    for arg in args:
        argname = f"{lib_prefix}{arg.name}"
        kwargs = dict(entries(arg.spec))
        parser.add_argument(argname, **kwargs)

    has_custom_help = "help" in argnames

    if f"{lib_prefix}help" in sys.argv and not has_custom_help:
        parser.print_help()
        sys.exit()

    args, extra = parser.parse_known_args()

    for item in extra:
        if item.startswith(lib_prefix):
            warnings.warn(
                f"Unrecognized argument {item!r} for {libname} (passed on as-is)"  # noqa: E501
            )
            break

    sys.argv = sys.argv[:1] + extra

    return args
