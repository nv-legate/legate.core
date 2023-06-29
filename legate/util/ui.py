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

from datetime import timedelta
from typing import Any, Iterable

from typing_extensions import TypeAlias

from .colors import bright, cyan, dim, green, magenta, red, white, yellow

Details: TypeAlias = Iterable[str]

__all__ = (
    "UI_WIDTH",
    "banner",
    "error",
    "failed",
    "key",
    "kvtable",
    "passed",
    "rule",
    "section",
    "skipped",
    "timeout",
    "value",
    "warn",
)


#: Width for terminal ouput headers and footers.
UI_WIDTH = 80


def _format_details(
    details: Iterable[str] | None = None, pre: str = "   "
) -> str:
    if details:
        return f"{pre}" + f"\n{pre}".join(f"{line}" for line in details)
    return ""


def banner(
    heading: str,
    *,
    char: str = "#",
    width: int = UI_WIDTH,
    details: Iterable[str] | None = None,
) -> str:
    """Generate a title banner, with optional details included.

    Parameters
    ----------
    heading : str
        Text to use for the title

    char : str, optional
        A character to use to frame the banner. (default: "#")

    width : int, optional
        How wide to draw the banner. (Note: user-supplied heading or
        details willnot be truncated if they exceed this width)

    details : Iterable[str], optional
        A list of lines to diplay inside the banner area below the heading

    """
    pre = f"{char*3} "
    divider = char * width
    if not details:
        return f"\n{divider}\n{pre}{heading}\n{divider}"
    return f"""
{divider}
{pre}
{pre}{heading}
{pre}
{_format_details(details, pre)}
{pre}
{divider}"""


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


def skipped(msg: str) -> str:
    """Report a skipped test with a cyan [SKIP]

    Parameters
    ----------
    msg : str
        Text to display after [SKIP]

    """
    return f"{cyan('[SKIP]')} {msg}"


def timeout(msg: str) -> str:
    """Report a timed-out test with a yellow [TIME]

    Parameters
    ----------
    msg : str
        Text to display after [TIME]

    """
    return f"{yellow('[TIME]')} {msg}"


def failed(
    msg: str, *, details: Details | None = None, exit_code: int | None = None
) -> str:
    """Report a failed test result with a bright red [FAIL].

    Parameters
    ----------
    msg : str
        Text to display after [FAIL]

    details : Iterable[str], optional
        A sequenece of text lines to diplay below the ``msg`` line

    """
    fail = f"{bright(red('[FAIL]'))}"
    exit = f"{bright(white(f' (exit: {exit_code}) '))}" if exit_code else ""
    if details:
        return f"{fail} {msg}{exit}\n{_format_details(details)}"
    return f"{fail} {msg}{exit}"


def passed(msg: str, *, details: Details | None = None) -> str:
    """Report a passed test result with a bright green [PASS].

    Parameters
    ----------
    msg : str
        Text to display after [PASS]

    details : Iterable[str], optional
        A sequenece of text lines to diplay below the ``msg`` line

    """
    if details:
        return f"{bright(green('[PASS]'))} {msg}\n{_format_details(details)}"
    return f"{bright(green('[PASS]'))} {msg}"


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


def rule(
    text: str | None = None,
    *,
    pad: int = 0,
    char: str = "-",
    N: int = UI_WIDTH,
) -> str:
    """Format a horizontal rule, optionally with text

    Parameters
    ----------
    text : str or None, optional
        If not None, display this text inline in the rule (default: None)

    pad : int, optional
        An amount of padding to put in front of the rule

    char: str, optional
        A character to use for the rule (default: "-")

    N : int, optional
        Character width for the rule (default: 80)

    Returns
    -------
        str

    """
    width = N - pad
    if text is None:
        return cyan(f"{char*width: >{N}}")
    return cyan(" " * pad + char * 3 + f"{f' {text} ' :{char}<{width-3}}")


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


def shell(cmd: str, *, char: str = "+") -> str:
    """Report a shell command in a dim white color.

    Parameters
    ----------
    cmd : str
        The shell command string to display

    char : str, optional
        A character to prefix the ``cmd`` with. (default: "+")

    """
    return dim(white(f"{char}{cmd}"))


def summary(
    name: str,
    total: int,
    passed: int,
    time: timedelta,
    *,
    justify: bool = True,
) -> str:
    """Generate a test result summary line.

    The output is bright green if all tests passed, otherwise bright red.

    Parameters
    ----------
    name : str
        A name to display in this summary line.

    total : int
        The total number of tests to report.

    passed : int
        The number of passed tests to report.

    time : timedelta
        The time taken to run the tests

    """
    summary = (
        f"{name}: Passed {passed} of {total} tests ({passed/total*100:0.1f}%) "
        f"in {time.total_seconds():0.2f}s"
        if total > 0
        else f"{name}: 0 tests are running, Please check"
    )
    color = green if passed == total and total > 0 else red
    return bright(color(f"{summary: >{UI_WIDTH}}" if justify else summary))


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
