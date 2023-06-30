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

from legate.tester import (
    DEFAULT_CPUS_PER_NODE,
    DEFAULT_GPU_DELAY,
    DEFAULT_GPU_MEMORY_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_NUMAMEM,
    DEFAULT_OMPS_PER_NODE,
    DEFAULT_OMPTHREADS,
    args as m,
)


class TestParserDefaults:
    def test_featurs(self) -> None:
        assert m.parser.get_default("features") is None

    def test_files(self) -> None:
        assert m.parser.get_default("files") is None

    def test_unit(self) -> None:
        assert m.parser.get_default("unit") is False

    def test_last_failed(self) -> None:
        assert m.parser.get_default("last_failed") is False

    def test_cpus(self) -> None:
        assert m.parser.get_default("cpus") == DEFAULT_CPUS_PER_NODE

    def test_gpus(self) -> None:
        assert m.parser.get_default("gpus") == DEFAULT_GPUS_PER_NODE

    def test_cpu_pin(self) -> None:
        assert m.parser.get_default("cpu_pin") == "partial"

    def test_gpu_delay(self) -> None:
        assert m.parser.get_default("gpu_delay") == DEFAULT_GPU_DELAY

    def test_fbmem(self) -> None:
        assert m.parser.get_default("fbmem") == DEFAULT_GPU_MEMORY_BUDGET

    def test_omps(self) -> None:
        assert m.parser.get_default("omps") == DEFAULT_OMPS_PER_NODE

    def test_ompthreads(self) -> None:
        assert m.parser.get_default("ompthreads") == DEFAULT_OMPTHREADS

    def test_numamem(self) -> None:
        assert m.parser.get_default("numamem") == DEFAULT_NUMAMEM

    def test_timeout(self) -> None:
        assert m.parser.get_default("timeout") is None

    def test_legate_dir(self) -> None:
        assert m.parser.get_default("legate_dir") is None

    def test_test_root(self) -> None:
        assert m.parser.get_default("test_root") is None

    def test_workers(self) -> None:
        assert m.parser.get_default("workers") is None

    def test_verbose(self) -> None:
        assert m.parser.get_default("verbose") == 0

    def test_dry_run(self) -> None:
        assert m.parser.get_default("dry_run") is False

    def test_debug(self) -> None:
        assert m.parser.get_default("debug") is False


class TestParserConfig:
    def test_parser_epilog(self) -> None:
        assert (
            m.parser.epilog
            == "Any extra arguments will be forwarded to the Legate script"
        )

    def test_parser_description(self) -> None:
        assert m.parser.description == "Run the Cunumeric test suite"
