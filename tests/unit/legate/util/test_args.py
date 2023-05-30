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

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable, TypeVar

import pytest
from pytest_mock import MockerFixture

import legate.util.args as m
import legate.util.info as info

from ...util import powerset

T = TypeVar("T")


class TestMultipleChoices:
    @pytest.mark.parametrize("choices", ([1, 2, 3], range(4), ("a", "b")))
    def test_init(self, choices: Iterable[T]) -> None:
        mc = m.MultipleChoices(choices)
        assert mc._choices == set(choices)

    def test_contains_item(self) -> None:
        choices = [1, 2, 3]
        mc = m.MultipleChoices(choices)
        for item in choices:
            assert item in mc

    def test_contains_subset(self) -> None:
        choices = [1, 2, 3]
        mc = m.MultipleChoices(choices)
        for subset in powerset(choices):
            assert subset in mc

    def test_iter(self) -> None:
        choices = [1, 2, 3]
        mc = m.MultipleChoices(choices)
        assert list(mc) == choices


class TestExtendAction:
    parser = ArgumentParser()
    parser.add_argument(
        "--foo", dest="foo", action=m.ExtendAction, choices=("a", "b", "c")
    )

    def test_single(self) -> None:
        ns = self.parser.parse_args(["--foo", "a"])
        assert ns.foo == ["a"]

    def test_multi(self) -> None:
        ns = self.parser.parse_args(["--foo", "a", "--foo", "b"])
        assert sorted(ns.foo) == ["a", "b"]

    def test_repeat(self) -> None:
        ns = self.parser.parse_args(["--foo", "a", "--foo", "a"])
        assert ns.foo == ["a"]


class TestInfoAction:
    parser = ArgumentParser()
    parser.add_argument("--info", action=m.InfoAction)

    def tests_basic(self, mocker: MockerFixture) -> None:
        mock_print_info = mocker.patch.object(info, "print_build_info")
        mock_sys_exit = mocker.patch("sys.exit")

        self.parser.parse_args(["--info"])

        mock_print_info.assert_called_once()
        mock_sys_exit.assert_called_once()


@dataclass(frozen=True)
class _TestObj:
    a: int = 10
    b: m.NotRequired[int] = m.Unset
    c: m.NotRequired[str] = "foo"
    d: m.NotRequired[str] = m.Unset


class TestArgSpec:
    def test_default(self) -> None:
        spec = m.ArgSpec("dest")
        assert spec.dest == "dest"
        assert spec.action == m.Unset

        # all others are unset
        assert set(m.entries(spec)) == {
            ("dest", "dest"),
        }


class TestArgument:
    def test_kwargs(self) -> None:
        arg = m.Argument("arg", m.ArgSpec("dest", default=2, help="help"))

        assert arg.kwargs == dict(m.entries(arg.spec))


def test_entries() -> None:
    assert set(m.entries(_TestObj())) == {("a", 10), ("c", "foo")}


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
