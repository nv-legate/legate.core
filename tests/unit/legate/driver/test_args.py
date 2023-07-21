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

from argparse import SUPPRESS

import legate.driver.args as m
import legate.driver.defaults as defaults


class TestParserDefaults:
    def test_allow_abbrev(self) -> None:
        assert not m.parser.allow_abbrev

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

    # binding

    def test_cpu_bind(self) -> None:
        assert m.parser.get_default("cpu_bind") is None

    def test_gpu_bind(self) -> None:
        assert m.parser.get_default("gpu_bind") is None

    def test_mem_bind(self) -> None:
        assert m.parser.get_default("mem_bind") is None

    def test_nic_bind(self) -> None:
        assert m.parser.get_default("nic_bind") is None

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

    # profiling

    def test_profile(self) -> None:
        assert m.parser.get_default("profile") is False

    def test_nvprof(self) -> None:
        assert m.parser.get_default("nvprof") is False

    def test_nsys(self) -> None:
        assert m.parser.get_default("nsys") is False

    def test_nsys_targets(self) -> None:
        assert (
            m.parser.get_default("nsys_targets")
            == "cublas,cuda,cudnn,nvtx,ucx"
        )

    def test_nsys_extra(self) -> None:
        assert m.parser.get_default("nsys_extra") == []

    # logging

    def test_logging(self) -> None:
        assert m.parser.get_default("logging") is None

    def test_logdir(self) -> None:
        assert m.parser.get_default("logdir") == defaults.LEGATE_LOG_DIR

    def test_log_to_file(self) -> None:
        assert m.parser.get_default("log_to_file") is False

    # debugging

    def test_gdb(self) -> None:
        assert m.parser.get_default("gdb") is False

    def test_cuda_gdb(self) -> None:
        assert m.parser.get_default("cuda_gdb") is False

    def test_memcheck(self) -> None:
        assert m.parser.get_default("memcheck") is False

    def test_freeze_on_error(self) -> None:
        assert m.parser.get_default("freeze_on_error") is False

    def test_gasnet_trace(self) -> None:
        assert m.parser.get_default("gasnet_trace") is False

    def test_dataflow(self) -> None:
        assert m.parser.get_default("dataflow") is False

    def test_event(self) -> None:
        assert m.parser.get_default("event") is False

    # info

    def test_progress(self) -> None:
        assert m.parser.get_default("progress") is False

    def test_mem_usage(self) -> None:
        assert m.parser.get_default("mem_usage") is False

    def test_verbose(self) -> None:
        assert m.parser.get_default("verbose") is False

    def test_bind_detail(self) -> None:
        assert m.parser.get_default("bind_detail") is False

    # other

    def test_module(self) -> None:
        assert m.parser.get_default("module") is None

    def test_dry_run(self) -> None:
        assert m.parser.get_default("dry_run") is False

    def test_rlwrap(self) -> None:
        assert m.parser.get_default("rlwrap") is False

    def test_info(self) -> None:
        assert m.parser.get_default("info") == SUPPRESS


class TestParserConfig:
    def test_parser_epilog(self) -> None:
        assert m.parser.epilog is None

    def test_parser_description(self) -> None:
        assert m.parser.description == "Legate Driver"
