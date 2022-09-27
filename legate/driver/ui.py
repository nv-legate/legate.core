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

import re
import sys
from typing import Any, Iterable

__all__ = (
    "bright",
    "cyan",
    "dim",
    "error",
    "green",
    "key",
    "kvtable",
    "magenta",
    "red",
    "rule",
    "scrub",
    "section",
    "value",
    "warn",
    "white",
    "yellow",
)


def _text(text: str) -> str:
    return text


try:
    import colorama  # type: ignore[import]

    def bright(text: str) -> str:
        return f"{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}"

    def dim(text: str) -> str:
        return f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"

    def white(text: str) -> str:
        return f"{colorama.Fore.WHITE}{text}{colorama.Style.RESET_ALL}"

    def cyan(text: str) -> str:
        return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"

    def red(text: str) -> str:
        return f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"

    def magenta(text: str) -> str:
        return f"{colorama.Fore.MAGENTA}{text}{colorama.Style.RESET_ALL}"

    def green(text: str) -> str:
        return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"

    def yellow(text: str) -> str:
        return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"

    if sys.platform == "win32":
        colorama.init()

except ImportError:

    bright = dim = white = cyan = red = magenta = green = yellow = _text

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


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


def scrub(text: str) -> str:
    """Remove ANSI color codes from a text string.

    Parameters
    ----------
    text : str
        The text to scrub

    Returns
    -------
        str

    """
    return _ANSI_ESCAPE.sub("", text)


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
