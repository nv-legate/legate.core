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

from datetime import timedelta
from typing import Any

import pytest
from pytest_mock import MockerFixture
from typing_extensions import TypeAlias

from legate.util import colors, ui as m

try:
    import colorama  # type: ignore [import-untyped]
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
    mocker.patch.object(m, "yellow", colors._text)
    mocker.patch.object(m, "magenta", colors._text)


def test_UI_WIDTH() -> None:
    assert m.UI_WIDTH == 80


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_banner_simple() -> None:
    assert (
        m.banner("some text")
        == "\n" + "#" * m.UI_WIDTH + "\n### some text\n" + "#" * m.UI_WIDTH
    )


def test_banner_simple_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert (
        m.banner("some text")
        == "\n" + "#" * m.UI_WIDTH + "\n### some text\n" + "#" * m.UI_WIDTH
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
def test_error() -> None:
    assert m.error("some message") == colors.red("ERROR: some message")


def test_error_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.error("some message") == "ERROR: some message"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_key() -> None:
    assert m.key("some key") == colors.dim(colors.green("some key"))


def test_key_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.key("some key") == "some key"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_value() -> None:
    assert m.value("some value") == m.yellow("some value")


def test_value_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.value("some value") == "some value"


class Test_kvtable:
    ONE = {"foo": 10}
    TWO = {"foo": 10, "barbaz": "some value"}
    THREE = {"foo": 10, "barbaz": "some value", "a": 1.2}

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_default(self, items: dict[str, Any]) -> None:
        N = max(len(m.key(k)) for k in items)
        assert m.kvtable(items) == "\n".join(
            f"{m.key(k): <{N}} : {m.value(str(items[k]))}" for k in items
        )

    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_default_plain(
        self, use_plain_text: UsePlainTextFixture, items: dict[str, Any]
    ) -> None:
        N = max(len(k) for k in items)
        assert m.kvtable(items) == "\n".join(
            f"{k: <{N}} : {items[k]}" for k in items
        )

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_delim(self, items: dict[str, Any]) -> None:
        N = max(len(m.key(k)) for k in items)
        assert m.kvtable(items, delim="/") == "\n".join(
            f"{m.key(k): <{N}}/{m.value(str(items[k]))}" for k in items
        )

    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_delim_plain(
        self, use_plain_text: UsePlainTextFixture, items: dict[str, Any]
    ) -> None:
        N = max(len(k) for k in items)
        assert m.kvtable(items, delim="/") == "\n".join(
            f"{k: <{N}}/{items[k]}" for k in items
        )

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_align_False(self, items: dict[str, Any]) -> None:
        assert m.kvtable(items, align=False) == "\n".join(
            f"{m.key(k)} : {m.value(str(items[k]))}" for k in items
        )

    @pytest.mark.parametrize("items", (ONE, TWO, THREE))
    def test_align_False_plain(
        self, use_plain_text: UsePlainTextFixture, items: dict[str, Any]
    ) -> None:
        assert m.kvtable(items, align=False) == "\n".join(
            f"{k} : {items[k]}" for k in items
        )

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_keys(self) -> None:
        items = self.THREE
        keys = ("foo", "a")
        N = max(len(m.key(k)) for k in items)

        assert m.kvtable(self.THREE, keys=keys) == "\n".join(
            f"{m.key(k): <{N}} : {m.value(str(items[k]))}" for k in keys
        )

    def test_keys_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        items = self.THREE
        keys = ("foo", "a")
        N = max(len(m.key(k)) for k in items)

        assert m.kvtable(items, keys=keys) == "\n".join(
            f"{k: <{N}} : {items[k]}" for k in keys
        )


class Test_rule:
    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_pad(self) -> None:
        assert m.rule(pad=4) == colors.cyan("    " + "-" * (m.UI_WIDTH - 4))

    def test_pad_with_text(
        self,
    ) -> None:
        front = "    --- foo bar "
        assert m.rule("foo bar", pad=4) == colors.cyan(
            front + "-" * (m.UI_WIDTH - len(front))
        )

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_text(self) -> None:
        front = "--- foo bar "
        assert m.rule("foo bar") == colors.cyan(
            front + "-" * (m.UI_WIDTH - len(front))
        )

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_char(self) -> None:
        assert m.rule(char="a") == colors.cyan("a" * m.UI_WIDTH)

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_N(self) -> None:
        assert m.rule(N=60) == colors.cyan("-" * 60)

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_N_with_text(self) -> None:
        front = "--- foo bar "
        assert m.rule("foo bar", N=65) == colors.cyan(
            front + "-" * (65 - len(front))
        )

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_pad_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        assert m.rule(pad=4) == "    " + "-" * (m.UI_WIDTH - 4)

    def test_pad_with_text_plain(
        self, use_plain_text: UsePlainTextFixture
    ) -> None:
        front = "    --- foo bar "
        assert m.rule("foo bar", pad=4) == front + "-" * (
            m.UI_WIDTH - len(front)
        )

    def test_text_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        front = "--- foo bar "
        assert m.rule("foo bar") == "--- foo bar " + "-" * (
            m.UI_WIDTH - len(front)
        )

    def test_char_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        assert m.rule(char="a") == "a" * m.UI_WIDTH

    def test_N_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        assert m.rule(N=60) == "-" * 60

    def test_N_with_text_plain(
        self, use_plain_text: UsePlainTextFixture
    ) -> None:
        front = "--- foo bar "
        assert m.rule("foo bar", N=65) == front + "-" * (65 - len(front))


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_section() -> None:
    assert m.section("some section") == m.bright(m.white("some section"))


def test_section_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.section("some section") == "some section"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_warn() -> None:
    assert m.warn("some message") == m.magenta("WARNING: some message")


def test_warn_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.warn("some message") == "WARNING: some message"


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
def test_failed_with_exit_code() -> None:
    fail = colors.bright(colors.red("[FAIL]"))
    exit = colors.bright(colors.white(" (exit: 10) "))
    assert m.failed("msg", exit_code=10) == f"{fail} msg{exit}"  # noqa


def test_failed_with_exit_code_plain(
    use_plain_text: UsePlainTextFixture,
) -> None:
    assert m.failed("msg", exit_code=10) == "[FAIL] msg (exit: 10) "


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
def test_failed_with_details_and_exit_code() -> None:
    fail = colors.bright(colors.red("[FAIL]"))
    exit = colors.bright(colors.white(" (exit: 10) "))
    assert (
        m.failed("msg", details=["a", "b"], exit_code=10)
        == f"{fail} msg{exit}\n   a\n   b"
    )


def test_failed_with_details_and_exit_code_plain(
    use_plain_text: UsePlainTextFixture,
) -> None:
    assert (
        m.failed("msg", details=["a", "b"], exit_code=10)
        == "[FAIL] msg (exit: 10) \n   a\n   b"
    )


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_skipped() -> None:
    assert m.skipped("msg") == f"{colors.cyan('[SKIP]')} msg"


def test_skipped_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.skipped("msg") == "[SKIP] msg"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_timeout() -> None:
    assert m.timeout("msg") == f"{colors.yellow('[TIME]')} msg"


def test_timeout_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert m.timeout("msg") == "[TIME] msg"


@pytest.mark.skipif(colorama is None, reason="colorama required")
def test_summary() -> None:
    assert m.summary("foo", 12, 11, timedelta(seconds=2.123)) == colors.bright(
        colors.red(
            f"{'foo: Passed 11 of 12 tests (91.7%) in 2.12s': >{m.UI_WIDTH}}"
        )
    )


def test_summary_plain(use_plain_text: UsePlainTextFixture) -> None:
    assert (
        m.summary("foo", 12, 11, timedelta(seconds=2.123))
        == f"{'foo: Passed 11 of 12 tests (91.7%) in 2.12s': >{m.UI_WIDTH}}"
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
