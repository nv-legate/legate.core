# Copyright AS2022 NVIDIA Corporation
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
"""Helpler functions for simple text UI output.

The color functions in this module require ``colorama`` to be installed in
order to generate color output. If ``colorama`` is not available, plain
text output (i.e. without ANSI color codes) will generated.

"""
from __future__ import annotations

import re
import sys
from typing import Any, Iterable

__all__ = (
    "bright",
    "cyan",
    "dim",
    "green",
    "red",
    "scrub",
    "white",
    "yellow",
)

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

    def green(text: str) -> str:
        return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"

    def yellow(text: str) -> str:
        return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"

    if sys.platform == "win32":
        colorama.init()

except ImportError:

    def _text(text: str) -> str:
        return text

    bright = dim = white = cyan = red = green = yellow = _text

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def key(text: str) -> str:
    return dim(green(text))


def value(text: str) -> str:
    return yellow(text)


def kvtable(
    kvs: dict[str, Any],
    *,
    delim: str = " : ",
    align: bool = True,
    keys: Iterable[str] | None = None,
) -> str:
    # annoying but necessary to take len on color-formatted version
    N = max(len(dim(green(name))) for name in kvs) if align else 0
    keys = kvs.keys() if keys is None else keys
    return "\n".join(
        f"{key(k): <{N}}{delim}{value(str(kvs[k]))}" for k in keys
    )


def rule(text: str | None = None, N: int = 80) -> str:
    if text is None:
        return cyan(f"{'-':-<{N}}")
    return cyan(f"{f'--- {text} ' :-<{N}}")


def section(text: str) -> str:
    return bright(white(text))


def scrub(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def warn(text: str) -> str:
    return red(f"WARNING: {text}")
