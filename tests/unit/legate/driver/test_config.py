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

import os
from pathlib import Path
from unittest.mock import call

import pytest
from pytest_mock import MockerFixture

import legate.driver.config as m
import legate.driver.defaults as defaults
from legate.util import colors
from legate.util.colors import scrub
from legate.util.types import DataclassMixin

from ...util import Capsys, powerset, powerset_nonempty

DEFAULTS_ENV_VARS = (
    "LEGATE_EAGER_ALLOC_PERCENTAGE",
    "LEGATE_FBMEM",
    "LEGATE_NUMAMEM",
    "LEGATE_OMP_PROCS",
    "LEGATE_OMP_THREADS",
    "LEGATE_REGMEM",
    "LEGATE_SYSMEM",
    "LEGATE_UTILITY_CORES",
    "LEGATE_ZCMEM",
)


class TestMultiNode:
    def test_fields(self) -> None:
        assert set(m.MultiNode.__dataclass_fields__) == {
            "nodes",
            "ranks_per_node",
            "not_control_replicable",
            "launcher",
            "launcher_extra",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.MultiNode, DataclassMixin)

    @pytest.mark.parametrize(
        "extra",
        (["a"], ["a", "b c"], ["a", "b c", "d e"], ["a", "b c", "d e", "f"]),
    )
    def test_launcher_extra_fixup_basic(self, extra: list[str]) -> None:
        mn = m.MultiNode(
            nodes=1,
            ranks_per_node=1,
            not_control_replicable=False,
            launcher="mpirun",
            launcher_extra=extra,
        )
        assert mn.launcher_extra == sum((x.split() for x in extra), [])

    def test_launcher_extra_fixup_complex(self) -> None:
        mn = m.MultiNode(
            nodes=1,
            ranks_per_node=1,
            not_control_replicable=False,
            launcher="mpirun",
            launcher_extra=[
                "-H g0002,g0002 -X SOMEENV --fork",
                "-bind-to none",
            ],
        )
        assert mn.launcher_extra == [
            "-H",
            "g0002,g0002",
            "-X",
            "SOMEENV",
            "--fork",
            "-bind-to",
            "none",
        ]

    def test_launcher_extra_fixup_quoted(self) -> None:
        mn = m.MultiNode(
            nodes=1,
            ranks_per_node=1,
            not_control_replicable=False,
            launcher="mpirun",
            launcher_extra=[
                "-f 'some path with spaces/foo.txt'",
            ],
        )
        assert mn.launcher_extra == [
            "-f",
            "some path with spaces/foo.txt",
        ]


class TestBinding:
    def test_fields(self) -> None:
        assert set(m.Binding.__dataclass_fields__) == {
            "cpu_bind",
            "mem_bind",
            "gpu_bind",
            "nic_bind",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Binding, DataclassMixin)


class TestCore:
    def test_fields(self) -> None:
        assert set(m.Core.__dataclass_fields__) == {
            "cpus",
            "gpus",
            "openmp",
            "ompthreads",
            "utility",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Core, DataclassMixin)


class TestMemory:
    def test_fields(self) -> None:
        assert set(m.Memory.__dataclass_fields__) == {
            "sysmem",
            "numamem",
            "fbmem",
            "zcmem",
            "regmem",
            "eager_alloc",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Memory, DataclassMixin)


class TestProfiling:
    def test_fields(self) -> None:
        assert set(m.Profiling.__dataclass_fields__) == {
            "profile",
            "cprofile",
            "nvprof",
            "nsys",
            "nsys_targets",
            "nsys_extra",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Profiling, DataclassMixin)

    @pytest.mark.parametrize(
        "extra",
        (["a"], ["a", "b c"], ["a", "b c", "d e"], ["a", "b c", "d e", "f"]),
    )
    def test_nsys_extra_fixup_basic(self, extra: list[str]) -> None:
        p = m.Profiling(
            profile=True,
            cprofile=True,
            nvprof=True,
            nsys=True,
            nsys_targets="foo,bar",
            nsys_extra=extra,
        )
        assert p.nsys_extra == sum((x.split() for x in extra), [])

    def test_nsys_extra_fixup_complex(self) -> None:
        p = m.Profiling(
            profile=True,
            cprofile=True,
            nvprof=True,
            nsys=True,
            nsys_targets="foo,bar",
            nsys_extra=[
                "-H g0002,g0002 -X SOMEENV --fork",
                "-bind-to none",
            ],
        )
        assert p.nsys_extra == [
            "-H",
            "g0002,g0002",
            "-X",
            "SOMEENV",
            "--fork",
            "-bind-to",
            "none",
        ]

    def test_nsys_extra_fixup_quoted(self) -> None:
        p = m.Profiling(
            profile=True,
            cprofile=True,
            nvprof=True,
            nsys=True,
            nsys_targets="foo,bar",
            nsys_extra=[
                "-f 'some path with spaces/foo.txt'",
            ],
        )
        assert p.nsys_extra == [
            "-f",
            "some path with spaces/foo.txt",
        ]


class TestLogging:
    def test_fields(self) -> None:
        assert set(m.Logging.__dataclass_fields__) == {
            "user_logging_levels",
            "logdir",
            "log_to_file",
            "keep_logs",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Logging, DataclassMixin)


class TestDebugging:
    def test_fields(self) -> None:
        assert set(m.Debugging.__dataclass_fields__) == {
            "gdb",
            "cuda_gdb",
            "memcheck",
            "valgrind",
            "freeze_on_error",
            "gasnet_trace",
            "dataflow",
            "event",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Debugging, DataclassMixin)


class TestInfo:
    def test_fields(self) -> None:
        assert set(m.Info.__dataclass_fields__) == {
            "progress",
            "mem_usage",
            "verbose",
            "bind_detail",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Info, DataclassMixin)


class TestOther:
    def test_fields(self) -> None:
        assert set(m.Other.__dataclass_fields__) == {
            "module",
            "dry_run",
            "rlwrap",
        }

    def test_mixin(self) -> None:
        assert issubclass(m.Other, DataclassMixin)


class TestConfig:
    def test_default_init(self) -> None:
        # Note this test does not clear the environment. Default values from
        # the defaults module can depend on the environment, but what matters
        # is that the generated config matches those values, whatever they are.

        c = m.Config(["legate"])

        assert colors.ENABLED is False

        assert c.multi_node == m.MultiNode(
            nodes=defaults.LEGATE_NODES,
            ranks_per_node=defaults.LEGATE_RANKS_PER_NODE,
            not_control_replicable=False,
            launcher="none",
            launcher_extra=[],
        )
        assert c.binding == m.Binding(
            cpu_bind=None,
            mem_bind=None,
            gpu_bind=None,
            nic_bind=None,
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
            logdir=Path(os.getcwd()),
            log_to_file=False,
            keep_logs=False,
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
        )

        assert c.info == m.Info(
            progress=False, mem_usage=False, verbose=False, bind_detail=False
        )

        assert c.other == m.Other(module=None, dry_run=False, rlwrap=False)

    def test_color_arg(self) -> None:
        m.Config(["legate", "--color"])

        assert colors.ENABLED is True

    def test_arg_conversions(self, mocker: MockerFixture) -> None:
        # This is kind of a dumb short-cut test, but if we believe that
        # object_to_dataclass works as advertised, then this test ensures that
        # it is being used for all the sub-configs that it should be used for

        spy = mocker.spy(m, "object_to_dataclass")

        c = m.Config(["legate"])

        assert spy.call_count == 9
        spy.assert_has_calls(
            [
                call(c._args, m.MultiNode),
                call(c._args, m.Binding),
                call(c._args, m.Core),
                call(c._args, m.Memory),
                call(c._args, m.Profiling),
                call(c._args, m.Logging),
                call(c._args, m.Debugging),
                call(c._args, m.Info),
                call(c._args, m.Other),
            ]
        )

    def test_nocr_fixup_default_single_node(self, capsys: Capsys) -> None:
        c = m.Config(["legate"])

        assert c.console
        assert not c.multi_node.not_control_replicable

        out, _ = capsys.readouterr()
        assert scrub(out).strip() == ""

    def test_nocr_fixup_multi_node(self, capsys: Capsys) -> None:
        c = m.Config(["legate", "--nodes", "2"])

        assert c.console
        assert c.multi_node.not_control_replicable

        out, _ = capsys.readouterr()
        assert (
            scrub(out).strip()
            == "WARNING: Disabling control replication for interactive run"
        )

    def test_nocr_fixup_multi_rank(self, capsys: Capsys) -> None:
        c = m.Config(["legate", "--ranks-per-node", "2"])

        assert c.console
        assert c.multi_node.not_control_replicable

        out, _ = capsys.readouterr()
        assert (
            scrub(out).strip()
            == "WARNING: Disabling control replication for interactive run"
        )

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    def test_log_to_file_fixup(
        self, capsys: Capsys, args: tuple[str, ...]
    ) -> None:
        # add --no-replicate to suppress unrelated stdout warning
        c = m.Config(
            ["legate", "--no-replicate", "--logging", "foo"] + list(args)
        )

        assert c.logging.log_to_file

        out, _ = capsys.readouterr()
        assert scrub(out).strip() == (
            "WARNING: Logging output is being redirected to a file in "
            f"directory {c.logging.logdir}"
        )

    # maybe this is overkill but this is literally the point where the user's
    # own script makes contact with legate, so let's make extra sure that that
    # ingest succeeds over a very wide range of command line combinations (one
    # option from most sub-configs)
    @pytest.mark.parametrize(
        "args",
        powerset(
            (
                "--no-replicate",
                "--rlwrap",
                "--dataflow",
                "--progress",
                "--gdb",
                "--keep-logs",
                "--profile",
                "--cprofile",
            )
        ),
    )
    def test_user_opts(self, args: tuple[str, ...]) -> None:
        c = m.Config(["legate"] + list(args) + ["foo.py", "-a", "1"])

        assert c.user_opts == ("-a", "1")
        assert c.user_script == "foo.py"

    def test_console_true(self) -> None:
        c = m.Config(["legate"])

        assert c.user_opts == ()
        assert c.console

    def test_console_false(self) -> None:
        c = m.Config(["legate", "--rlwrap", "--gpus", "2", "foo.py", "-a"])

        assert c.user_opts == ("-a",)
        assert c.user_script == "foo.py"
        assert not c.console
