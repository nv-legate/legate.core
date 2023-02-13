# Copyright 2023 NVIDIA Corporation
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

import pytest

import legate.settings as m
from legate.util.settings import PrioritizedSetting

_expected_settings = (
    "consensus",
    "cycle_check",
    "future_leak_check",
)


class TestSettings:
    def test_standard_settings(self) -> None:
        settings = [
            k
            for k, v in m.settings.__class__.__dict__.items()
            if isinstance(v, PrioritizedSetting)
        ]
        assert set(settings) == set(_expected_settings)

    @pytest.mark.parametrize("name", _expected_settings)
    def test_prefix(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert ps.env_var.startswith("LEGATE_")

    @pytest.mark.parametrize("name", _expected_settings)
    def test_parent(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert ps._parent == m.settings

    def test_types(self) -> None:
        assert m.settings.consensus.convert_type == "bool"
        assert m.settings.cycle_check.convert_type == "bool"
        assert m.settings.future_leak_check.convert_type == "bool"


class TestDefaults:
    def test_consensus(self) -> None:
        assert m.settings.consensus.default is False

    def test_cycle_check(self) -> None:
        assert m.settings.cycle_check.default is False

    def test_future_leak_check(self) -> None:
        assert m.settings.future_leak_check.default is False
