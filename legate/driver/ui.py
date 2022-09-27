# Copyright 2022 NVIDIA Corporation
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
"""Helper functions for simple text UI output.

The color functions in this module require ``colorama`` to be installed in
order to generate color output. If ``colorama`` is not available, plain
text output (i.e. without ANSI color codes) will be generated.

"""
from __future__ import annotations

from typing import Any, Iterable

from ..utils.colors import (
    bright,
    cyan,
    dim,
    green,
    magenta,
    red,
    white,
    yellow,
)

__all__ = (
    "error",
    "key",
    "kvtable",
    "rule",
    "section",
    "value",
    "warn",
)


def error(text: str) -> str:
    """Format text as an error.

    Parameters
    ----------
    text : str
        The text to format

    Returns
    -------
        str

    """
    return red(f"ERROR: {text}")


def key(text: str) -> str:
    """Format a 'key' from a key-value pair.

    Parameters
    ----------
    text : str
        The key to format

    Returns
    -------
        str

    """
    return dim(green(text))


def value(text: str) -> str:
    """Format a 'value' from of a key-value pair.

    Parameters
    ----------
    text : str
        The key to format

    Returns
    -------
        str

    """
    return yellow(text)


def kvtable(
    items: dict[str, Any],
    *,
    delim: str = " : ",
    align: bool = True,
    keys: Iterable[str] | None = None,
) -> str:
    """Format a dictionay as a table of key-value pairs.

    Parameters
    ----------
    items : dict[str, Any]
        The dictionary of items to format

    delim : str, optional
        A delimiter to display between keys and values (default: " : ")

    align : bool, optional
        Whether to align delimiters to the longest key length (default: True)

    keys : Iterable[str] or None, optional
        If not None, only the specified subset of keys is included in the
        table output (default: None)

    Returns
    -------
        str

    """
    # annoying but necessary to take len on color-formatted version
    N = max(len(key(k)) for k in items) if align else 0

    keys = items.keys() if keys is None else keys

    return "\n".join(
        f"{key(k): <{N}}{delim}{value(str(items[k]))}" for k in keys
    )


def rule(text: str | None = None, *, char: str = "-", N: int = 80) -> str:
    """Format a horizontal rule, optionally with text

    Parameters
    ----------
    text : str or None, optional
        If not None, display this text inline in the rule (default: None)

    char: str, optional
        A character to use for the rule (default: "-")

    N : int, optional
        Character width for the rule (default: 80)

    Returns
    -------
        str

    """
    if text is None:
        return cyan(char * N)
    return cyan(char * 3 + f"{f' {text} ' :{char}<{N-3}}")


def section(text: str) -> str:
    """Format text as a section header

    Parameters
    ----------
    text : str
        The text to format

    Returns
    -------
        str

    """
    return bright(white(text))


def warn(text: str) -> str:
    """Format text as a warning.

    Parameters
    ----------
    text : str
        The text to format

    Returns
    -------
        str

    """
    return magenta(f"WARNING: {text}")
