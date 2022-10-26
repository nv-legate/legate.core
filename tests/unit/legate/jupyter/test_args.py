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

import legate.driver.args as m
import legate.driver.defaults as defaults


class TestParserDefaults:
    def test_allow_abbrev(self) -> None:
        assert not m.parser.allow_abbrev

    # kernel

    def test_no_user(self) -> None:
        assert m.parser.get_default("user") is None

    def test_name(self) -> None:
        assert m.parser.get_default("name") is None

    def test_display_name(self) -> None:
        assert m.parser.get_default("display_name") is None

    def test_prefix(self) -> None:
        assert m.parser.get_default("prefix") is None

    # multi_node

    def test_nodes(self) -> None:
        assert m.parser.get_default("nodes") == defaults.LEGATE_NODES

    def test_ranks_per_node(self) -> None:
        assert (
            m.parser.get_default("ranks_per_node")
            == defaults.LEGATE_RANKS_PER_NODE
        )

    def test_no_replicate(self) -> None:
        assert m.parser.get_default("not_control_replicable") is False

    def test_launcher(self) -> None:
        assert m.parser.get_default("launcher") == "none"

    def test_launcher_extra(self) -> None:
        assert m.parser.get_default("launcher_extra") == []

    # core

    def test_cpus(self) -> None:
        assert m.parser.get_default("cpus") == defaults.LEGATE_CPUS

    def test_gpus(self) -> None:
        assert m.parser.get_default("gpus") == defaults.LEGATE_GPUS

    def test_omps(self) -> None:
        assert m.parser.get_default("openmp") == defaults.LEGATE_OMP_PROCS

    def test_ompthreads(self) -> None:
        assert (
            m.parser.get_default("ompthreads") == defaults.LEGATE_OMP_THREADS
        )

    def test_utility(self) -> None:
        assert m.parser.get_default("utility") == defaults.LEGATE_UTILITY_CORES

    # memory

    def test_sysmem(self) -> None:
        assert m.parser.get_default("sysmem") == defaults.LEGATE_SYSMEM

    def test_numamem(self) -> None:
        assert m.parser.get_default("numamem") == defaults.LEGATE_NUMAMEM

    def test_fbmem(self) -> None:
        assert m.parser.get_default("fbmem") == defaults.LEGATE_FBMEM

    def test_zcmem(self) -> None:
        assert m.parser.get_default("zcmem") == defaults.LEGATE_ZCMEM

    def test_regmem(self) -> None:
        assert m.parser.get_default("regmem") == defaults.LEGATE_REGMEM

    def test_eager_alloc(self) -> None:
        assert (
            m.parser.get_default("eager_alloc")
            == defaults.LEGATE_EAGER_ALLOC_PERCENTAGE
        )

    # info

    def test_verbose(self) -> None:
        assert m.parser.get_default("verbose") is False
