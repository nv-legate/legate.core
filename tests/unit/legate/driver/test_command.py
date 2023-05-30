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

import argparse
import sys
from pathlib import Path

import pytest

import legate.driver.command as m
from legate import install_info
from legate.driver.launcher import RANK_ENV_VARS
from legate.util.colors import scrub
from legate.util.types import LauncherType

from ...util import Capsys, powerset_nonempty
from .util import GenObjs


def test___all__() -> None:
    assert m.__all__ == (
        "CMD_PARTS_LEGION",
        "CMD_PARTS_CANONICAL",
    )


def test_LEGATE_GLOBAL_RANK_SUBSTITUTION() -> None:
    assert m.LEGATE_GLOBAL_RANK_SUBSTITUTION == "%%LEGATE_GLOBAL_RANK%%"


def test_CMD_PARTS() -> None:
    assert m.CMD_PARTS_LEGION == (
        m.cmd_bind,
        m.cmd_rlwrap,
        m.cmd_gdb,
        m.cmd_cuda_gdb,
        m.cmd_nvprof,
        m.cmd_nsys,
        m.cmd_memcheck,
        m.cmd_valgrind,
        m.cmd_legion,
        m.cmd_python_processor,
        m.cmd_module,
        m.cmd_nocr,
        m.cmd_kthreads,
        m.cmd_cpus,
        m.cmd_gpus,
        m.cmd_openmp,
        m.cmd_utility,
        m.cmd_bgwork,
        m.cmd_mem,
        m.cmd_numamem,
        m.cmd_fbmem,
        m.cmd_regmem,
        m.cmd_network,
        m.cmd_log_levels,
        m.cmd_log_file,
        m.cmd_eager_alloc,
        m.cmd_user_script,
        m.cmd_user_opts,
    )


class Test_cmd_bind:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (bind_sh, "--launcher", "local", "--")

    def test_bind_detail(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--bind-detail"])

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (bind_sh, "--launcher", "local", "--debug", "--")

    @pytest.mark.parametrize("kind", ("cpu", "gpu", "mem", "nic"))
    def test_basic_local(self, genobjs: GenObjs, kind: str) -> None:
        config, system, launcher = genobjs([f"--{kind}-bind", "1"])

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (
            bind_sh,
            "--launcher",
            "local",
            f"--{kind}s",
            "1",
            "--",
        )

    @pytest.mark.parametrize("launch", ("none", "mpirun", "jsrun", "srun"))
    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_combo_local(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
    ) -> None:
        all_binds = [
            "--cpu-bind",
            "1",
            "--gpu-bind",
            "1",
            "--nic-bind",
            "1",
            "--mem-bind",
            "1",
            "--launcher",
            launch,
        ]
        config, system, launcher = genobjs(all_binds)

        result = m.cmd_bind(config, system, launcher)

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result[:3] == (
            bind_sh,
            "--launcher",
            "local" if launch == "none" else launch,
        )
        x = iter(result[3:])
        for name, binding in zip(x, x):  # pairwise
            assert f"{name} {binding}" in "--cpus 1 --gpus 1 --nics 1 --mems 1"

        assert result[-1] == "--"

    @pytest.mark.parametrize("launch", ("none", "mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("kind", ("cpu", "gpu", "mem", "nic"))
    def test_ranks_good(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
        kind: str,
        rank_var: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(install_info, "networks", ["ucx"])

        config, system, launcher = genobjs(
            [f"--{kind}-bind", "1/2", "--launcher", launch],
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )

        result = m.cmd_bind(config, system, launcher)

        launcher_arg = "auto" if launch == "none" else launch

        bind_sh = str(system.legate_paths.bind_sh_path)
        assert result == (
            bind_sh,
            "--launcher",
            launcher_arg,
            f"--{kind}s",
            "1/2",
            "--",
        )

    @pytest.mark.parametrize("binding", ("1", "1/2/3"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("kind", ("cpu", "gpu", "mem", "nic"))
    def test_ranks_bad(
        self,
        genobjs: GenObjs,
        binding: str,
        kind: str,
        rank_var: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(install_info, "networks", ["ucx"])

        config, system, launcher = genobjs(
            [f"--{kind}-bind", binding],
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )

        msg = (
            f"Number of groups in --{kind}-bind not equal to --ranks-per-node"
        )
        with pytest.raises(RuntimeError, match=msg):
            m.cmd_bind(config, system, launcher)

    @pytest.mark.parametrize("launch", ("none", "mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_no_networking_error(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
        rank_var: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(install_info, "networks", [])

        config, system, launcher = genobjs(
            ["--launcher", launch],
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )

        msg = (
            "multi-rank run was requested, but Legate was not built with "
            "networking support"
        )
        with pytest.raises(RuntimeError, match=msg):
            m.cmd_bind(config, system, launcher)


class Test_cmd_gdb:
    MULTI_RANK_WARN = (
        "WARNING: Legate does not support gdb for multi-rank runs"
    )

    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_gdb(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option(self, genobjs: GenObjs, os: str) -> None:
        config, system, launcher = genobjs(["--gdb"], os=os)

        result = m.cmd_gdb(config, system, launcher)

        debugger = ("lldb", "--") if os == "Darwin" else ("gdb", "--args")
        assert result == debugger

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option_multi_rank(
        self, genobjs: GenObjs, capsys: Capsys, os: str, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--gdb"], multi_rank=(2, 2), rank_env={rank_var: "1"}, os=os
        )

        result = m.cmd_gdb(config, system, launcher)
        assert result == ()

        out, _ = capsys.readouterr()
        assert scrub(out).strip() == self.MULTI_RANK_WARN


class Test_cmd_cuda_gdb:
    MULTI_RANK_WARN = (
        "WARNING: Legate does not support cuda-gdb for multi-rank runs"
    )

    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_cuda_gdb(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option(self, genobjs: GenObjs, os: str) -> None:
        config, system, launcher = genobjs(["--cuda-gdb"], os=os)

        result = m.cmd_cuda_gdb(config, system, launcher)

        assert result == ("cuda-gdb", "--args")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("os", ("Darwin", "Linux"))
    def test_with_option_multi_rank(
        self, genobjs: GenObjs, capsys: Capsys, os: str, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--cuda-gdb"], multi_rank=(2, 2), rank_env={rank_var: "1"}, os=os
        )

        result = m.cmd_cuda_gdb(config, system, launcher)
        assert result == ()

        out, _ = capsys.readouterr()
        assert scrub(out).strip() == self.MULTI_RANK_WARN


class Test_cmd_nvprof:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_nvprof(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--nvprof", "--logdir", "foo"])

        result = m.cmd_nvprof(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
        )
        assert result == ("nvprof", "-o", log_path)

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nvprof", "--logdir", "foo"],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nvprof(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
        )
        assert result == ("nvprof", "-o", log_path)

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_multi_rank_with_launcher(
        self,
        genobjs: GenObjs,
        launch: str,
    ) -> None:
        config, system, launcher = genobjs(
            ["--nvprof", "--logdir", "foo", "--launcher", launch],
            multi_rank=(2, 2),
        )

        result = m.cmd_nvprof(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
        )
        assert result == ("nvprof", "-o", log_path)


class Test_cmd_nsys:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_nsys(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nsys", "--logdir", "foo"],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == (
            "nsys",
            "profile",
            "-t",
            "cublas,cuda,cudnn,nvtx,ucx",
            "-o",
            log_path,
            "-s",
            "none",
        )

    @pytest.mark.parametrize(
        "nsys_extra",
        (
            ["--nsys-extra=--sample=cpu"],
            ["--nsys-extra", "--backtrace=lbr -s cpu"],
            ["--nsys-extra", "--sample cpu"],
            ["--nsys-extra", "-s cpu"],
            ["--nsys-extra=--sample", "--nsys-extra", "cpu"],
        ),
    )
    def test_explicit_sample(
        self, genobjs: GenObjs, nsys_extra: list[str]
    ) -> None:
        args = ["--nsys"] + nsys_extra
        config, system, launcher = genobjs(args)
        result = m.cmd_nsys(config, system, launcher)

        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--sample")
        parsed_args, unparsed = parser.parse_known_args(result)

        assert parsed_args.sample == "cpu"

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_multi_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--nsys", "--logdir", "foo", "--launcher", launch],
            multi_rank=(2, 2),
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == (
            "nsys",
            "profile",
            "-t",
            "cublas,cuda,cudnn,nvtx,ucx",
            "-o",
            log_path,
            "-s",
            "none",
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_extra_no_s(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            [
                "--nsys",
                "--logdir",
                "foo",
                "--nsys-extra",
                "a",
                "--nsys-extra",
                "b",
            ],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == (
            "nsys",
            "profile",
            "-t",
            "cublas,cuda,cudnn,nvtx,ucx",
            "-o",
            log_path,
            "a",
            "b",
            "-s",
            "none",
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_extra_with_s(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            [
                "--nsys",
                "--logdir",
                "foo",
                "--nsys-extra",
                "a",
                "--nsys-extra",
                "b",
                "--nsys-extra=-s",  # note have to use "=" format
                "--nsys-extra",
                "foo",
            ],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == (
            "nsys",
            "profile",
            "-t",
            "cublas,cuda,cudnn,nvtx,ucx",
            "-o",
            log_path,
            "a",
            "b",
            "-s",
            "foo",
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_multi_rank_targets(
        self, genobjs: GenObjs, rank_var: str, rank: str
    ) -> None:
        config, system, launcher = genobjs(
            [
                "--nsys",
                "--logdir",
                "foo",
                "--nsys-targets",
                "foo,bar",
            ],
            multi_rank=(2, 2),
            rank_env={rank_var: rank},
        )

        result = m.cmd_nsys(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}"
        )
        assert result == (
            "nsys",
            "profile",
            "-t",
            "foo,bar",
            "-o",
            log_path,
            "-s",
            "none",
        )


class Test_cmd_memcheck:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_memcheck(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--memcheck"])

        result = m.cmd_memcheck(config, system, launcher)

        assert result == ("compute-sanitizer",)


class Test_cmd_valgrind:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_valgrind(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--valgrind"])

        result = m.cmd_valgrind(config, system, launcher)

        assert result == ("valgrind",)


class Test_cmd_nocr:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_nocr(config, system, launcher)

        assert result == ()

    def test_console_single_node(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([], fake_module=None)

        result = m.cmd_nocr(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_console_multi_node(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            # passing --nodes is not usually necessary for genobjs but we
            # are probing a "fixup" check that inspect args directly
            ["--nodes", "2"],
            multi_rank=(2, 1),
            rank_env={rank_var: rank},
            fake_module=None,
        )

        result = m.cmd_nocr(config, system, launcher)

        assert result == ("--nocr",)

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_console_multi_rank(
        self, genobjs: GenObjs, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            # passing --ranks-per-node is not usually necessary for genobjs
            # but we are probing a "fixup" check that inspect args directly
            ["--ranks-per-node", "2"],
            multi_rank=(1, 2),
            rank_env={rank_var: "0"},
            fake_module=None,
        )

        result = m.cmd_nocr(config, system, launcher)

        assert result == ("--nocr",)

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--no-replicate"])

        result = m.cmd_nocr(config, system, launcher)

        assert result == ("--nocr",)


class Test_cmd_module:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_module(config, system, launcher)

        assert result == ()

    def test_with_module(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--module", "foo"])

        result = m.cmd_module(config, system, launcher)

        assert result == ("-m", "foo")

    def test_with_cprofile(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--cprofile"])

        result = m.cmd_module(config, system, launcher)

        log_path = str(
            config.logging.logdir
            / f"legate_{m.LEGATE_GLOBAL_RANK_SUBSTITUTION}.cprof"
        )
        assert result == ("-m", "cProfile", "-o", log_path)

    def test_module_and_cprofile_error(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--module", "foo", "--cprofile"])

        err = "Only one of --module or --cprofile may be used"
        with pytest.raises(ValueError, match=err):
            m.cmd_module(config, system, launcher)


class Test_cmd_rlwrap:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_rlwrap(config, system, launcher)

        assert result == ()

    def test_with_option(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--rlwrap"])

        result = m.cmd_rlwrap(config, system, launcher)

        assert result == ("rlwrap",)


class Test_cmd_legion:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_legion(config, system, launcher)

        assert result == (str(system.legion_paths.legion_python),)


class Test_cmd_python_processor:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_python_processor(config, system, launcher)

        assert result == ("-ll:py", "1")


class Test_cmd_kthreads:
    DBG_OPTS = ("--gdb", "--cuda-gdb", "--freeze-on-error")

    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_kthreads(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("args", powerset_nonempty(DBG_OPTS), ids=str)
    def test_with_debugging(self, genobjs: GenObjs, args: list[str]) -> None:
        config, system, launcher = genobjs(list(args))

        result = m.cmd_kthreads(config, system, launcher)

        assert result == ("-ll:force_kthreads",)


class Test_cmd_cpus:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_cpus(config, system, launcher)

        assert result == ("-ll:cpu", str(config.core.cpus))

    def test_one(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--cpus", "1"])

        result = m.cmd_cpus(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("value", ("2", "16"))
    def test_multiple(self, genobjs: GenObjs, value: str) -> None:
        config, system, launcher = genobjs(["--cpus", value])

        result = m.cmd_cpus(config, system, launcher)

        assert result == ("-ll:cpu", value)


class Test_cmd_gpus:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_gpus(config, system, launcher)

        assert result == ()

    def test_zero(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--gpus", "0"])

        result = m.cmd_gpus(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("value", ("1", "2", "16"))
    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_nonzero(self, genobjs: GenObjs, value: str) -> None:
        config, system, launcher = genobjs(["--gpus", value])

        result = m.cmd_gpus(config, system, launcher)

        assert result == ("-ll:gpu", value, "-cuda:skipbusy")

    def test_without_cuda(
        self, genobjs: GenObjs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(install_info, "use_cuda", False)

        config, system, launcher = genobjs(["--gpus", "1"])

        with pytest.raises(
            RuntimeError,
            match="--gpus was requested, but this build does not have CUDA "
            "support enabled",
        ):
            m.cmd_gpus(config, system, launcher)


class Test_cmd_openmp:
    ZERO_THREAD_WARN = (
        "WARNING: Legate is ignoring request for "
        "{omps} OpenMP processors with 0 threads"
    )

    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_openmp(config, system, launcher)

        assert result == ()

    def test_omps_zero(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--omps", "0"])

        result = m.cmd_openmp(config, system, launcher)

        assert result == ()

    def test_without_openmp(
        self, genobjs: GenObjs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(install_info, "use_openmp", False)

        config, system, launcher = genobjs(["--omps", "1"])

        with pytest.raises(
            RuntimeError,
            match="--omps was requested, but this build does not have OpenMP "
            "support enabled",
        ):
            m.cmd_openmp(config, system, launcher)

    @pytest.mark.parametrize("omps", ("1", "12"))
    @pytest.mark.skipif(
        not install_info.use_openmp, reason="no OpenMP support"
    )
    def test_ompthreads_zero(
        self, genobjs: GenObjs, capsys: Capsys, omps: str
    ) -> None:
        args = ["--ompthreads", "0", "--omps", omps]
        config, system, launcher = genobjs(args)

        result = m.cmd_openmp(config, system, launcher)

        assert result == ()

        out, _ = capsys.readouterr()
        assert scrub(out).strip() == self.ZERO_THREAD_WARN.format(omps=omps)

    @pytest.mark.parametrize("omps", ("1", "2", "12"))
    @pytest.mark.parametrize("ompthreads", ("1", "2", "12"))
    @pytest.mark.skipif(
        not install_info.use_openmp, reason="no OpenMP support"
    )
    def test_ompthreads_no_numa(
        self, genobjs: GenObjs, omps: str, ompthreads: str
    ) -> None:
        args = ["--ompthreads", ompthreads, "--omps", omps]
        config, system, launcher = genobjs(args)

        result = m.cmd_openmp(config, system, launcher)

        assert result == (
            "-ll:ocpu",
            omps,
            "-ll:othr",
            ompthreads,
            "-ll:onuma",
            "0",
        )

    @pytest.mark.parametrize("omps", ("1", "2", "12"))
    @pytest.mark.parametrize("ompthreads", ("1", "2", "12"))
    @pytest.mark.skipif(
        not install_info.use_openmp, reason="no OpenMP support"
    )
    def test_ompthreads_with_numa(
        self, genobjs: GenObjs, omps: str, ompthreads: str
    ) -> None:
        args = ["--ompthreads", ompthreads, "--omps", omps, "--numamem", "100"]
        config, system, launcher = genobjs(args)

        result = m.cmd_openmp(config, system, launcher)

        assert result == (
            "-ll:ocpu",
            omps,
            "-ll:othr",
            ompthreads,
            "-ll:onuma",
            "1",
        )


class Test_cmd_utility:
    def test_default_single_rank(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_utility(config, system, launcher)

        assert result == ("-ll:util", "2")

    def test_utility_1_single_rank(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--utility", "1"])

        result = m.cmd_utility(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utiltity_n_single_rank(
        self, genobjs: GenObjs, value: str
    ) -> None:
        config, system, launcher = genobjs(["--utility", value])

        result = m.cmd_utility(config, system, launcher)

        assert result == ("-ll:util", value)

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_default_multi_rank(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            [], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        result = m.cmd_utility(config, system, launcher)

        assert result == ("-ll:util", "2")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_utility_1_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", "1"], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        result = m.cmd_utility(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_utility_1_multi_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", "1", "--launcher", launch], multi_rank=(2, 2)
        )

        result = m.cmd_utility(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utility_n_multi_rank_no_launcher(
        self, genobjs: GenObjs, value: str, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", value], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        result = m.cmd_utility(config, system, launcher)

        assert result == ("-ll:util", value)

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utility_n_multi_rank_with_launcher(
        self, genobjs: GenObjs, value: str, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", value, "--launcher", launch], multi_rank=(2, 2)
        )

        result = m.cmd_utility(config, system, launcher)

        assert result == ("-ll:util", value)


class Test_cmd_bgwork:
    def test_default_single_rank(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ()

    def test_utility_1_single_rank(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--utility", "1"])

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ()

    def test_utility_1_single_rank_and_ucx(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--utility", "1"])

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ()

    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utiltity_n_single_rank(
        self, genobjs: GenObjs, value: str
    ) -> None:
        config, system, launcher = genobjs(["--utility", value])

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utiltity_n_single_rank_and_ucx(
        self, genobjs: GenObjs, value: str
    ) -> None:
        config, system, launcher = genobjs(["--utility", value])

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ()

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_default_multi_rank(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            [], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ("-ll:bgwork", "2")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_default_multi_rank_and_ucx(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            [], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ("-ll:bgwork", "2", "-ll:bgworkpin", "1")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_utility_1_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", "1"], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ("-ll:bgwork", "2")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    def test_utility_1_multi_rank_no_launcher_and_ucx(
        self, genobjs: GenObjs, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", "1"], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ("-ll:bgwork", "2", "-ll:bgworkpin", "1")

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_utility_1_multi_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", "1", "--launcher", launch], multi_rank=(2, 2)
        )

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ("-ll:bgwork", "2")

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_utility_1_multi_rank_with_launcher_and_ucx(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", "1", "--launcher", launch], multi_rank=(2, 2)
        )

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ("-ll:bgwork", "2", "-ll:bgworkpin", "1")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utility_n_multi_rank_no_launcher(
        self, genobjs: GenObjs, value: str, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", value], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ("-ll:bgwork", value)

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.parametrize("rank", ("0", "1", "2"))
    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utility_n_multi_rank_no_launcher_and_ucx(
        self, genobjs: GenObjs, value: str, rank: str, rank_var: dict[str, str]
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", value], multi_rank=(2, 2), rank_env={rank_var: rank}
        )

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ("-ll:bgwork", value, "-ll:bgworkpin", "1")

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utility_n_multi_rank_with_launcher(
        self, genobjs: GenObjs, value: str, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", value, "--launcher", launch], multi_rank=(2, 2)
        )

        result = m.cmd_bgwork(config, system, launcher)

        assert result == ("-ll:bgwork", value)

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("value", ("2", "3", "10"))
    def test_utility_n_multi_rank_with_launcher_and_ucx(
        self, genobjs: GenObjs, value: str, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--utility", value, "--launcher", launch], multi_rank=(2, 2)
        )

        networks_orig = list(install_info.networks)
        install_info.networks.append("ucx")
        result = m.cmd_bgwork(config, system, launcher)
        install_info.networks[:] = networks_orig[:]

        assert result == ("-ll:bgwork", value, "-ll:bgworkpin", "1")


class Test_cmd_sysmem:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_mem(config, system, launcher)

        assert result == ("-ll:csize", str(config.memory.sysmem))

    @pytest.mark.parametrize("value", ("0", "100", "12344"))
    def test_value(self, genobjs: GenObjs, value: str) -> None:
        config, system, launcher = genobjs(["--sysmem", value])

        result = m.cmd_mem(config, system, launcher)

        assert result == ("-ll:csize", value)


class Test_cmd_numamem:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_numamem(config, system, launcher)

        assert result == ()

    def test_zero(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--numamem", "0"])

        result = m.cmd_numamem(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("value", ("100", "12344"))
    def test_nonzero(self, genobjs: GenObjs, value: str) -> None:
        config, system, launcher = genobjs(["--numamem", value])

        result = m.cmd_numamem(config, system, launcher)

        assert result == ("-ll:nsize", value)


class Test_cmd_fbmem:
    @pytest.mark.parametrize("fb", ("10", "1234"))
    @pytest.mark.parametrize("zc", ("10", "1234"))
    @pytest.mark.parametrize("gpus", ([], ["--gpus", "0"]), ids=str)
    def test_no_gpus_with_values(
        self, genobjs: GenObjs, gpus: list[str], fb: str, zc: str
    ) -> None:
        config, system, launcher = genobjs(
            gpus + ["--fbmem", fb, "--zcmem", zc]
        )

        result = m.cmd_fbmem(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("gpus", ([], ["--gpus", "0"]), ids=str)
    def test_no_gpus_no_values(
        self, genobjs: GenObjs, gpus: list[str]
    ) -> None:
        config, system, launcher = genobjs(gpus)

        result = m.cmd_fbmem(config, system, launcher)

        assert result == ()

    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_with_gpus_no_values(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--gpus", "1"])

        result = m.cmd_fbmem(config, system, launcher)

        assert result == (
            "-ll:fsize",
            str(config.memory.fbmem),
            "-ll:zsize",
            str(config.memory.zcmem),
        )  # defaults

    @pytest.mark.parametrize("fb", ("10", "1234"))
    @pytest.mark.parametrize("zc", ("10", "1234"))
    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_with_gpus(self, genobjs: GenObjs, fb: str, zc: str) -> None:
        args = ["--gpus", "1", "--fbmem", fb, "--zcmem", zc]
        config, system, launcher = genobjs(args)

        result = m.cmd_fbmem(config, system, launcher)

        assert result == ("-ll:fsize", fb, "-ll:zsize", zc)


class Test_cmd_regmem:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_regmem(config, system, launcher)

        assert result == ()

    def test_zero(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--regmem", "0"])

        result = m.cmd_regmem(config, system, launcher)

        assert result == ()

    @pytest.mark.parametrize("value", ("100", "12344"))
    def test_nonzero(self, genobjs: GenObjs, value: str) -> None:
        config, system, launcher = genobjs(["--regmem", value])

        result = m.cmd_regmem(config, system, launcher)

        assert result == ("-ll:rsize", value)


class Test_cmd_network:
    def test_no_launcher_single_rank(
        self,
        genobjs: GenObjs,
    ) -> None:
        config, system, launcher = genobjs()
        result = m.cmd_network(config, system, launcher)
        assert result == ("-ll:networks", "none")

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_no_launcher_multi_rank(
        self,
        genobjs: GenObjs,
        rank_var: dict[str, str],
    ) -> None:
        config, system, launcher = genobjs(
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )
        result = m.cmd_network(config, system, launcher)
        assert result == ()

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_launcher_single_rank(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
    ) -> None:
        config, system, launcher = genobjs(["--launcher", launch])
        result = m.cmd_network(config, system, launcher)
        assert result == ("-ll:networks", "none")

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_launcher_multi_rank(
        self,
        genobjs: GenObjs,
        launch: LauncherType,
    ) -> None:
        config, system, launcher = genobjs(
            ["--launcher", launch],
            multi_rank=(2, 2),
        )
        result = m.cmd_network(config, system, launcher)
        assert result == ()


class Test_cmd_log_levels:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_log_levels(config, system, launcher)

        assert result == ("-level", "openmp=5")

    def test_default_with_user_levels(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--logging", "foo,bar"])

        result = m.cmd_log_levels(config, system, launcher)

        assert result == ("-level", "openmp=5,foo,bar")

    def test_profile_single_rank_no_launcher(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--profile"])

        result = m.cmd_log_levels(config, system, launcher)

        log_file = str(config.logging.logdir / "legate_%.prof")
        assert result == (
            ("-lg:prof", "1")
            + ("-lg:prof_logfile", log_file)
            + ("-level", "openmp=5,legion_prof=2")
        )

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_profile_single_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(["--profile", "--launcher", launch])

        result = m.cmd_log_levels(config, system, launcher)

        log_file = str(config.logging.logdir / "legate_%.prof")
        assert result == (
            ("-lg:prof", "1")
            + ("-lg:prof_logfile", log_file)
            + ("-level", "openmp=5,legion_prof=2")
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_profile_multi_rank_no_launcher(
        self, genobjs: GenObjs, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--profile"], multi_rank=(2, 2), rank_env={rank_var: "2"}
        )

        result = m.cmd_log_levels(config, system, launcher)

        log_file = str(config.logging.logdir / "legate_%.prof")
        assert result == (
            ("-lg:prof", "4")
            + ("-lg:prof_logfile", log_file)
            + ("-level", "openmp=5,legion_prof=2")
        )

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_profile_multi_rank_with_launcher(
        self, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--profile", "--launcher", launch], multi_rank=(2, 2)
        )

        result = m.cmd_log_levels(config, system, launcher)

        log_file = str(config.logging.logdir / "legate_%.prof")
        assert result == (
            ("-lg:prof", "4")
            + ("-lg:prof_logfile", log_file)
            + ("-level", "openmp=5,legion_prof=2")
        )

    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_gpus(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--gpus", "2"])

        result = m.cmd_log_levels(config, system, launcher)

        assert result == ("-level", "openmp=5,gpu=5")

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    def test_debugging(self, genobjs: GenObjs, args: tuple[str, ...]) -> None:
        config, system, launcher = genobjs(list(args))

        result = m.cmd_log_levels(config, system, launcher)

        assert result == ("-lg:spy", "-level", "openmp=5,legion_spy=2")

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_combined(
        self, genobjs: GenObjs, launch: str, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            [
                "--profile",
                "--launcher",
                launch,
                "--profile",
                "--gpus",
                "4",
                "--logging",
                "foo",
            ],
            multi_rank=(2, 2),
            rank_env={rank_var: "2"},
        )

        result = m.cmd_log_levels(config, system, launcher)

        log_file = str(config.logging.logdir / "legate_%.prof")
        assert result == (
            ("-lg:prof", "4")
            + ("-lg:prof_logfile", log_file)
            + ("-level", "openmp=5,legion_prof=2,gpu=5,foo")
        )


class Test_cmd_log_file:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        result = m.cmd_log_file(config, system, launcher)

        assert result == ()

    def test_dir_without_flag(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--logdir", "foo"])

        result = m.cmd_log_file(config, system, launcher)

        assert result == ()

    def test_flag_without_dir(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(["--log-to-file"])

        result = m.cmd_log_file(config, system, launcher)

        logfile = str(config.logging.logdir / "legate_%.log")
        assert result == ("-logfile", logfile, "-errlevel", "4")

    def test_flag_with_dir(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs(
            ["--log-to-file", "--logdir", "foo"]
        )

        result = m.cmd_log_file(config, system, launcher)

        logfile = str(Path("foo") / "legate_%.log")
        assert result == ("-logfile", logfile, "-errlevel", "4")


class Test_cmd_eager_alloc:
    @pytest.mark.parametrize("value", ("0", "1", "12", "99", "100"))
    def test_basic(self, genobjs: GenObjs, value: str) -> None:
        config, system, launcher = genobjs(["--eager-alloc-percentage", value])

        result = m.cmd_eager_alloc(config, system, launcher)

        assert result == ("-lg:eager_alloc_percentage", value)


class Test_cmd_user_opts:
    USER_OPTS: tuple[list[str], ...] = (
        [],
        ["foo"],
        ["foo.py"],
        ["foo.py", "10"],
        ["foo.py", "--baz", "10"],
    )

    @pytest.mark.parametrize("opts", USER_OPTS, ids=str)
    def test_basic(self, genobjs: GenObjs, opts: list[str]) -> None:
        config, system, launcher = genobjs(opts, fake_module=None)

        user_opts = m.cmd_user_opts(config, system, launcher)
        user_script = m.cmd_user_script(config, system, launcher)
        result = user_script + user_opts

        assert result == tuple(opts)

    @pytest.mark.parametrize("opts", USER_OPTS, ids=str)
    @pytest.mark.skipif(not install_info.use_cuda, reason="no CUDA support")
    def test_with_legate_opts(self, genobjs: GenObjs, opts: list[str]) -> None:
        args = ["--verbose", "--rlwrap", "--gpus", "2"] + opts
        config, system, launcher = genobjs(args, fake_module=None)

        user_opts = m.cmd_user_opts(config, system, launcher)
        user_script = m.cmd_user_script(config, system, launcher)
        result = user_script + user_opts

        assert result == tuple(opts)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
