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

from pathlib import Path

import pytest

import legate.settings as m
from legate.util.fs import read_c_define
from legate.util.settings import EnvOnlySetting, PrioritizedSetting, _Unset

_expected_settings = (
    "consensus",
    "cycle_check",
    "future_leak_check",
    "test",
    "min_gpu_chunk",
    "min_cpu_chunk",
    "min_omp_chunk",
    "window_size",
    "max_pending_exceptions",
    "precise_exception_trace",
    "field_reuse_frac",
    "field_reuse_freq",
    "max_lru_length",
)

ENV_HEADER = Path(__file__).parents[3] / "src" / "env_defaults.h"


class TestSettings:
    def test_standard_settings(self) -> None:
        settings = [
            k
            for k, v in m.settings.__class__.__dict__.items()
            if isinstance(v, (PrioritizedSetting, EnvOnlySetting))
        ]
        assert set(settings) == set(_expected_settings)

    @pytest.mark.parametrize("name", _expected_settings)
    def test_prefix(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert ps.env_var.startswith("LEGATE_")

    def test_types(self) -> None:
        assert m.settings.consensus.convert_type == 'bool ("0" or "1")'
        assert m.settings.cycle_check.convert_type == 'bool ("0" or "1")'
        assert m.settings.future_leak_check.convert_type == 'bool ("0" or "1")'


_settings_with_test_defaults = (
    "min_gpu_chunk",
    "min_cpu_chunk",
    "min_omp_chunk",
    "window_size",
    # skipping due to LEGATE_DEBUG conditioning
    # "max_pending_exceptions",
    "precise_exception_trace",
    "field_reuse_frac",
    "field_reuse_freq",
    "max_lru_length",
)


class TestDefaults:
    def test_consensus(self) -> None:
        assert m.settings.consensus.default is False

    def test_cycle_check(self) -> None:
        assert m.settings.cycle_check.default is False

    def test_future_leak_check(self) -> None:
        assert m.settings.future_leak_check.default is False

    def test_test(self) -> None:
        assert m.settings.test.default is False
        assert m.settings.test.test_default is _Unset

    @pytest.mark.parametrize("name", _settings_with_test_defaults)
    def test_default(self, name: str) -> None:
        setting = getattr(m.settings, name)
        define = setting.env_var.removeprefix("LEGATE_") + "_DEFAULT"
        expected = setting._convert(read_c_define(ENV_HEADER, define))
        assert setting.default == expected

    @pytest.mark.parametrize("name", _settings_with_test_defaults)
    def test_test_default(self, name: str) -> None:
        setting = getattr(m.settings, name)
        define = setting.env_var.removeprefix("LEGATE_") + "_TEST"
        expected = setting._convert(read_c_define(ENV_HEADER, define))
        assert setting.test_default == expected
