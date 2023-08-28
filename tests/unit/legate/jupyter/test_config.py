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

from pathlib import Path
from unittest.mock import call

from pytest_mock import MockerFixture

import legate.driver.defaults as defaults
import legate.jupyter.config as m
from legate.driver.config import Core, Memory, MultiNode
from legate.util import colors
from legate.util.types import DataclassMixin


class TestKernel:
    def test_fields(self) -> None:
        assert set(m.Kernel.__dataclass_fields__) == {
            "user",
            "prefix",
            "spec_name",
            "display_name",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Kernel, DataclassMixin)


class TestConfig:
    def test_default_init(self) -> None:
        # Note this test does not clear the environment. Default values from
        # the defaults module can depend on the environment, but what matters
        # is that the generated config matches those values, whatever they are.

        c = m.Config(["legate-jupyter"])

        assert colors.ENABLED is False

        assert c.multi_node == m.MultiNode(
            nodes=defaults.LEGATE_NODES,
            ranks_per_node=defaults.LEGATE_RANKS_PER_NODE,
            not_control_replicable=False,
            launcher="none",
            launcher_extra=[],
        )
        assert c.core == m.Core(
            cpus=4,
            gpus=0,
            openmp=defaults.LEGATE_OMP_PROCS,
            ompthreads=defaults.LEGATE_OMP_THREADS,
            utility=defaults.LEGATE_UTILITY_CORES,
        )
        c.memory == m.Memory(
            sysmem=defaults.LEGATE_SYSMEM,
            numamem=defaults.LEGATE_NUMAMEM,
            fbmem=defaults.LEGATE_FBMEM,
            zcmem=defaults.LEGATE_ZCMEM,
            regmem=defaults.LEGATE_REGMEM,
            eager_alloc=defaults.LEGATE_EAGER_ALLOC_PERCENTAGE,
        )

        # These are all "turned off"

        assert c.binding == m.Binding(
            cpu_bind=None,
            mem_bind=None,
            gpu_bind=None,
            nic_bind=None,
        )

        c.profiling == m.Profiling(
            profile=False,
            cprofile=False,
            nvprof=False,
            nsys=False,
            nsys_targets="",
            nsys_extra=[],
        )

        assert c.logging == m.Logging(
            user_logging_levels=None,
            logdir=Path("."),
            log_to_file=False,
        )

        assert c.debugging == m.Debugging(
            gdb=False,
            cuda_gdb=False,
            memcheck=False,
            valgrind=False,
            freeze_on_error=False,
            gasnet_trace=False,
            dataflow=False,
            event=False,
            collective=False,
            spy_assert_warning=False,
        )

        assert c.info == m.Info(
            progress=False, mem_usage=False, verbose=False, bind_detail=False
        )

        assert c.other == m.Other(
            timing=False,
            wrapper=[],
            wrapper_inner=[],
            module=None,
            dry_run=False,
            rlwrap=False,
        )

    def test_color_arg(self) -> None:
        m.Config(["legate-jupyter", "--color"])

        assert colors.ENABLED is True

    def test_arg_conversions(self, mocker: MockerFixture) -> None:
        # This is kind of a dumb short-cut test, but if we believe that
        # object_to_dataclass works as advertised, then this test ensures that
        # it is being used for all the sub-configs that it should be used for

        spy = mocker.spy(m, "object_to_dataclass")

        c = m.Config(["legate"])

        assert spy.call_count == 4
        spy.assert_has_calls(
            [
                call(c._args, m.Kernel),
                call(c._args, MultiNode),
                call(c._args, Core),
                call(c._args, Memory),
            ]
        )
