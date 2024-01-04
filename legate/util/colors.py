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
"""Helper functions for adding colors to simple text UI output.

The color functions in this module require ``colorama`` to be installed in
order to generate color output. If ``colorama`` is not available, plain
text output (i.e. without ANSI color codes) will be generated.

"""
from __future__ import annotations

import re
import sys

__all__ = (
    "bright",
    "cyan",
    "dim",
    "green",
    "magenta",
    "red",
    "scrub",
    "white",
    "yellow",
)


# Color terminal output needs to be explicitly opt-in. Applications that want
# to enable it should set this global flag to True, e.g based on a command line
# argument or other user-supplied configuration
ENABLED = False


def _text(text: str) -> str:
    return text


try:
    import colorama  # type: ignore[import-untyped]

    def bright(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}"

    def dim(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"

    def white(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.WHITE}{text}{colorama.Style.RESET_ALL}"

    def cyan(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"

    def red(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"

    def magenta(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.MAGENTA}{text}{colorama.Style.RESET_ALL}"

    def green(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"

    def yellow(text: str) -> str:
        if not ENABLED:
            return text
        return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"

    if sys.platform == "win32":
        colorama.init()

except ImportError:
    bright = dim = white = cyan = red = magenta = green = yellow = _text

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


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
