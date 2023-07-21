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

import pytest
from pytest_mock import MockerFixture

import legate.driver.logs as m
from legate.driver.config import Config
from legate.driver.launcher import RANK_ENV_VARS
from legate.util.colors import scrub

from ...util import Capsys, powerset_nonempty
from .util import GenObjs


class MockHandler(m.LogHandler):
    _process_called = False

    def process(self) -> None:
        self._process_called = True


_EXPECTED_RANK_WARN = """\
WARNING: Skipping the processing of toolname output, to avoid wasting
resources in a large allocation. Please manually run: foo bar"""


class TestLogHandler:
    def test_init(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        handler = MockHandler(config, system)

        assert handler.config == config
        assert handler.system == system

    def test_run_processing_command_basic(
        self, mocker: MockerFixture, genobjs: GenObjs
    ) -> None:
        config, system, launcher = genobjs([])
        mock_run = mocker.patch.object(m, "run")

        handler = MockHandler(config, system)

        handler.run_processing_cmd(("foo", "bar"), "toolname")

        mock_run.assert_called_once_with(
            ("foo", "bar"), check=True, cwd=config.logging.logdir
        )

    def test_run_processing_command_verbose(
        self, capsys: Capsys, mocker: MockerFixture, genobjs: GenObjs
    ) -> None:
        config, system, launcher = genobjs(["--verbose"])
        mock_run = mocker.patch.object(m, "run")

        handler = MockHandler(config, system)

        handler.run_processing_cmd(("foo", "bar"), "toolname")

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == "Running: foo bar"

        mock_run.assert_called_once_with(
            ("foo", "bar"), check=True, cwd=config.logging.logdir
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_run_processing_command_mulit_rank_no_launcher_no_warning(
        self, mocker: MockerFixture, genobjs: GenObjs, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            [],
            multi_rank=(2, 2),
            rank_env={rank_var: "1"},
        )
        mock_run = mocker.patch.object(m, "run")

        handler = MockHandler(config, system)

        handler.run_processing_cmd(("foo", "bar"), "toolname")

        mock_run.assert_called_once_with(
            ("foo", "bar"), check=True, cwd=config.logging.logdir
        )

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_run_processing_command_mulit_rank_with_launcher_no_warning(
        self, mocker: MockerFixture, genobjs: GenObjs, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--launcher", launch], multi_rank=(2, 2)
        )
        mock_run = mocker.patch.object(m, "run")

        handler = MockHandler(config, system)

        handler.run_processing_cmd(("foo", "bar"), "toolname")

        mock_run.assert_called_once_with(
            ("foo", "bar"), check=True, cwd=config.logging.logdir
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_run_processing_command_mulit_rank_no_launcher_with_warning(
        self,
        capsys: Capsys,
        mocker: MockerFixture,
        genobjs: GenObjs,
        rank_var: str,
    ) -> None:
        config, system, launcher = genobjs(
            [],
            multi_rank=(5, 2),
            rank_env={rank_var: "1"},
        )
        mock_run = mocker.patch.object(m, "run")

        handler = MockHandler(config, system)

        handler.run_processing_cmd(("foo", "bar"), "toolname")

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == _EXPECTED_RANK_WARN
        mock_run.assert_not_called()

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_run_processing_command_mulit_rank_with_launcher_with_warning(
        self,
        capsys: Capsys,
        mocker: MockerFixture,
        genobjs: GenObjs,
        launch: str,
    ) -> None:
        config, system, launcher = genobjs(
            ["--launcher", launch], multi_rank=(5, 2)
        )
        mock_run = mocker.patch.object(m, "run")

        handler = MockHandler(config, system)

        handler.run_processing_cmd(("foo", "bar"), "toolname")

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == _EXPECTED_RANK_WARN
        mock_run.assert_not_called()


class Test_process_logs:
    def test_default(self, genobjs: GenObjs) -> None:
        config, system, launcher = genobjs([])

        with m.process_logs(config, system, launcher) as handlers:
            assert handlers == ()

    def test_with_profiling(
        self, mocker: MockerFixture, genobjs: GenObjs
    ) -> None:
        config, system, launcher = genobjs(["--profile"])

        mocker.patch.object(m, "ProfilingHandler", MockHandler)

        with m.process_logs(config, system, launcher) as handlers:
            pass

        assert len(handlers) == 1
        assert isinstance(handlers[0], MockHandler)
        assert handlers[0]._process_called

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    def test_with_debugging(
        self, mocker: MockerFixture, genobjs: GenObjs, args: tuple[str, ...]
    ) -> None:
        config, system, launcher = genobjs(list(args))

        mocker.patch.object(m, "DebuggingHandler", MockHandler)

        with m.process_logs(config, system, launcher) as handlers:
            pass

        assert len(handlers) == 1
        assert isinstance(handlers[0], MockHandler)
        assert handlers[0]._process_called

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    def test_with_debugging_and_profiling(
        self, mocker: MockerFixture, genobjs: GenObjs, args: tuple[str, ...]
    ) -> None:
        config, system, launcher = genobjs(["--profile"] + list(args))

        mocker.patch.object(m, "ProfilingHandler", MockHandler)
        mocker.patch.object(m, "DebuggingHandler", MockHandler)

        with m.process_logs(config, system, launcher) as handlers:
            pass

        assert len(handlers) == 2
        assert isinstance(handlers[0], MockHandler)
        assert handlers[0]._process_called
        assert isinstance(handlers[1], MockHandler)
        assert handlers[1]._process_called


def _de_flag(config: Config) -> tuple[str, ...]:
    dflag = "d" if config.debugging.dataflow else ""
    eflag = "e" if config.debugging.event else ""
    if dflag or eflag:
        return ("-{dflag}{eflag}",)
    return ()


class TestDebuggingHandler:
    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    def test_process_single_rank_no_launcher(
        self, genobjs: GenObjs, mocker: MockerFixture, args: str
    ) -> None:
        config, system, launcher = genobjs([])

        handler = m.DebuggingHandler(config, system)
        mock_run = mocker.patch.object(handler, "run_processing_cmd")

        handler.process()

        legion_spy_py = str(handler.system.legion_paths.legion_spy_py)
        expected = (legion_spy_py,) + _de_flag(config) + ("legate_0.log",)
        mock_run.assert_called_once_with(expected, "spy")

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_process_single_rank_with_launcher(
        self, genobjs: GenObjs, mocker: MockerFixture, launch: str, args: str
    ) -> None:
        config, system, launcher = genobjs(["--launcher", launch])

        handler = m.DebuggingHandler(config, system)
        mock_run = mocker.patch.object(handler, "run_processing_cmd")

        handler.process()

        legion_spy_py = str(handler.system.legion_paths.legion_spy_py)
        expected = (legion_spy_py,) + _de_flag(config) + ("legate_0.log",)
        mock_run.assert_called_once_with(expected, "spy")

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_process_multi_rank_no_launcher(
        self, genobjs: GenObjs, mocker: MockerFixture, rank_var: str, args: str
    ) -> None:
        config, system, launcher = genobjs(
            [], multi_rank=(3, 2), rank_env={rank_var: "1"}
        )

        handler = m.DebuggingHandler(config, system)
        mock_run = mocker.patch.object(handler, "run_processing_cmd")

        handler.process()

        legion_spy_py = str(handler.system.legion_paths.legion_spy_py)
        expected = (
            (legion_spy_py,)
            + _de_flag(config)
            + tuple(f"legate_{i}.log" for i in range(6))
        )
        mock_run.assert_called_once_with(expected, "spy")

    @pytest.mark.parametrize(
        "args", powerset_nonempty(("--event", "--dataflow"))
    )
    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_process_multi_rank_with_launcher(
        self, genobjs: GenObjs, mocker: MockerFixture, launch: str, args: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--launcher", launch], multi_rank=(3, 2)
        )

        handler = m.DebuggingHandler(config, system)
        mock_run = mocker.patch.object(handler, "run_processing_cmd")

        handler.process()

        legion_spy_py = str(handler.system.legion_paths.legion_spy_py)
        expected = (
            (legion_spy_py,)
            + _de_flag(config)
            + tuple(f"legate_{i}.log" for i in range(6))
        )
        mock_run.assert_called_once_with(expected, "spy")


class TestProcessingHandler:
    def test_process_single_rank_no_launcher(
        self, genobjs: GenObjs, capsys: Capsys
    ) -> None:
        config, system, launcher = genobjs([])

        handler = m.ProfilingHandler(config, system)

        handler.process()

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == (
            f"Profiles have been generated under {config.logging.logdir}, "
            f"run legion_prof --view "
            f"{config.logging.logdir}/legate_*.prof to view them"
        )

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_process_single_rank_with_launcher(
        self, genobjs: GenObjs, capsys: Capsys, launch: str
    ) -> None:
        config, system, launcher = genobjs(["--launcher", launch])

        handler = m.ProfilingHandler(config, system)

        handler.process()

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == (
            f"Profiles have been generated under {config.logging.logdir}, "
            f"run legion_prof --view "
            f"{config.logging.logdir}/legate_*.prof to view them"
        )

    @pytest.mark.parametrize("rank_var", RANK_ENV_VARS)
    def test_process_multi_rank_no_launcher(
        self, genobjs: GenObjs, capsys: Capsys, rank_var: str
    ) -> None:
        config, system, launcher = genobjs(
            [], multi_rank=(3, 2), rank_env={rank_var: "1"}
        )

        handler = m.ProfilingHandler(config, system)

        handler.process()

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == (
            f"Profiles have been generated under {config.logging.logdir}, "
            f"run legion_prof --view "
            f"{config.logging.logdir}/legate_*.prof to view them"
        )

    @pytest.mark.parametrize("launch", ("mpirun", "jsrun", "srun"))
    def test_process_multi_rank_with_launcher(
        self, genobjs: GenObjs, capsys: Capsys, launch: str
    ) -> None:
        config, system, launcher = genobjs(
            ["--launcher", launch], multi_rank=(3, 2)
        )

        handler = m.ProfilingHandler(config, system)

        handler.process()

        out, _ = capsys.readouterr()

        assert scrub(out).strip() == (
            f"Profiles have been generated under {config.logging.logdir}, "
            f"run legion_prof --view "
            f"{config.logging.logdir}/legate_*.prof to view them"
        )
