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

from typing import Any

import pytest
from pytest_mock import MockerFixture
from typing_extensions import TypeAlias

import legate.driver.ui as m
import legate.utils.colors as colors

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
    mocker.patch.object(m, "yellow", colors._text)
    mocker.patch.object(m, "magenta", colors._text)


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
    def test_text(self) -> None:
        assert m.rule("foo bar") == colors.cyan("--- foo bar " + "-" * 68)

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_char(self) -> None:
        assert m.rule(char="a") == colors.cyan("a" * 80)

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_N(self) -> None:
        assert m.rule(N=60) == colors.cyan("-" * 60)

    @pytest.mark.skipif(colorama is None, reason="colorama required")
    def test_N_with_text(self) -> None:
        assert m.rule("foo bar", N=65) == colors.cyan(
            "--- foo bar " + "-" * 53
        )

    def test_text_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        assert m.rule("foo bar") == "--- foo bar " + "-" * 68

    def test_char_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        assert m.rule(char="a") == "a" * 80

    def test_N_plain(self, use_plain_text: UsePlainTextFixture) -> None:
        assert m.rule(N=60) == "-" * 60

    def test_N_with_text_plain(
        self, use_plain_text: UsePlainTextFixture
    ) -> None:
        assert m.rule("foo bar", N=65) == "--- foo bar " + "-" * 53


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
