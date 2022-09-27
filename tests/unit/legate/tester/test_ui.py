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
"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from pytest_mock import MockerFixture
from typing_extensions import TypeAlias

import legate.utils.colors as colors
from legate.tester import UI_WIDTH, ui as m

try:
    import colorama  # type: ignore
except ImportError:
    colorama = None

UsePlainTextFixture: TypeAlias = Any


@pytest.fixture
def use_plain_text(mocker: MockerFixture) -> None:
    mocker.patch.object(m, "bright", colors._text)
    mocker.patch.object(m, "dim", colors._text)
    mocker.patch.object(m, "white", colors._text)
    mocker.patch.object(m, "cyan", colors._text)
    mocker.patch.object(m, "red", colors._text)
    mocker.patch.object(m, "green", colors._text)


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_banner_simple() -> None:
    assert (
        m.banner("some text")
        == "\n" + "#" * UI_WIDTH + "\n### some text\n" + "#" * UI_WIDTH
    )


def test_banner_simple_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert (
        m.banner("some text")
        == "\n" + "#" * UI_WIDTH + "\n### some text\n" + "#" * UI_WIDTH
    )


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_banner_full() -> None:
    assert (
        m.banner("some text", char="*", width=100, details=["a", "b"])
        == "\n"
        + "*" * 100
        + "\n*** \n*** some text\n*** \n*** a\n*** b\n*** \n"
        + "*" * 100
    )


def test_banner_full_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert (
        m.banner("some text", char="*", width=100, details=["a", "b"])
        == "\n"
        + "*" * 100
        + "\n*** \n*** some text\n*** \n*** a\n*** b\n*** \n"
        + "*" * 100
    )


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_rule_default() -> None:
    assert m.rule() == "    " + "~" * (UI_WIDTH - 4)


def test_rule_default_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.rule() == "    " + "~" * (UI_WIDTH - 4)


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_rule_with_args() -> None:
    assert m.rule(10, "-") == " " * 10 + "-" * (UI_WIDTH - 10)


def test_rule_with_args_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.rule(10, "-") == " " * 10 + "-" * (UI_WIDTH - 10)


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_shell() -> None:
    assert m.shell("cmd --foo") == colors.dim(colors.white("+cmd --foo"))


def test_shell_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.shell("cmd --foo") == "+cmd --foo"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_shell_with_char() -> None:
    assert m.shell("cmd --foo", char="") == colors.dim(
        colors.white("cmd --foo")
    )


def test_shell_with_char_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.shell("cmd --foo", char="") == "cmd --foo"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_passed() -> None:
    assert m.passed("msg") == f"{colors.bright(colors.green('[PASS]'))} msg"


def test_passed_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.passed("msg") == "[PASS] msg"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_passed_with_details() -> None:
    assert (
        m.passed("msg", details=["a", "b"])
        == f"{colors.bright(colors.green('[PASS]'))} msg\n   a\n   b"
    )


def test_passed_with_details_plain(
    use_plain_text: UsePlainTextFixture,
) -> None:
    assert m.passed("msg", details=["a", "b"]) == "[PASS] msg\n   a\n   b"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_failed() -> None:
    assert m.failed("msg") == f"{colors.bright(colors.red('[FAIL]'))} msg"


def test_failed_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.failed("msg") == "[FAIL] msg"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_failed_with_details() -> None:
    assert (
        m.failed("msg", details=["a", "b"])
        == f"{colors.bright(colors.red('[FAIL]'))} msg\n   a\n   b"
    )


def test_failed_with_details_plain(
    use_plain_text: UsePlainTextFixture,
) -> None:
    assert m.failed("msg", details=["a", "b"]) == "[FAIL] msg\n   a\n   b"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_skipped() -> None:
    assert m.skipped("msg") == f"{colors.cyan('[SKIP]')} msg"


def test_skipped_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.skipped("msg") == "[SKIP] msg"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_summary() -> None:
    assert m.summary("foo", 12, 11, timedelta(seconds=2.123)) == colors.bright(
        colors.red(
            f"{'foo: Passed 11 of 12 tests (91.7%) in 2.12s': >{UI_WIDTH}}"
        )
    )


def test_summary_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert (
        m.summary("foo", 12, 11, timedelta(seconds=2.123))
        == f"{'foo: Passed 11 of 12 tests (91.7%) in 2.12s': >{UI_WIDTH}}"
    )


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_summary_no_justify() -> None:
    assert m.summary(
        "foo", 12, 11, timedelta(seconds=2.123), justify=False
    ) == colors.bright(
        colors.red("foo: Passed 11 of 12 tests (91.7%) in 2.12s")
    )


def test_summary_no_justify_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert (
        m.summary("foo", 12, 11, timedelta(seconds=2.123), justify=False)
        == "foo: Passed 11 of 12 tests (91.7%) in 2.12s"
    )
