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
from shlex import quote
from subprocess import run
from textwrap import indent

from .command import CMD_PARTS
from .config import Config
from .launcher import Launcher
from .system import System
from .types import Command, EnvDict
from .ui import bright, cyan, dim, green, white, yellow

__all__ = ("Driver",)

_DARWIN_GDB_WARN = """\
WARNING: You must start the debugging session with the following command,
as LLDB no longer forwards the environment to subprocesses for security
reasons:

(lldb) process launch -v LIB_PATH={libpath} -v PYTHONPATH={pythonpath}

"""

_TOOL_WARN = """\
Skipping the processing of {tool} output, to avoid wasting resources in a
large allocation. Please manually run: {cmd}"
"""


class Driver:
    def __init__(self, config: Config, system: System) -> None:
        self.config = config
        self.system = system
        self.launcher = Launcher.create(config, system)

    @property
    def cmd(self) -> Command:
        config = self.config
        launcher = self.launcher
        system = self.system

        parts = (part(config, system, launcher) for part in CMD_PARTS)
        return launcher.cmd + sum(parts, ())

    @property
    def env(self) -> EnvDict:
        # in case we want to augment the launcher env we could do it here
        return self.launcher.env

    @property
    def custom_env_vars(self) -> set[str]:
        # in case we want to augment the launcher env we could do it here
        return self.launcher.custom_env_vars

    def run(self) -> int:
        if self.config.info.verbose:
            self._print_verbose()

        self._darwin_gdb_warn()

        if self.config.other.dry_run:
            return 0

        self._init_logging()

        proc = run(self.cmd, env=self.env)

        self._process_logging()

        return proc.returncode

    def _darwin_gdb_warn(self) -> None:
        gdb = self.config.debugging.gdb

        if gdb and self.system.os == "Darwin":
            libpath = self.env[self.system.LIB_PATH]
            pythonpath = self.env["PYTHONPATH"]
            print(
                _DARWIN_GDB_WARN.format(libpath=libpath, pythonpath=pythonpath)
            )

    def _init_logging(self) -> None:
        os.makedirs(self.config.logging.logdir, exist_ok=True)

    def _print_verbose(self) -> None:

        print(cyan(f"\n{'--- Legion Python Configuration ':-<80}"))

        print(bright(white("\nLegate paths:")))
        print(indent(str(self.system.legate_paths), prefix="  "))

        print(bright(white("\nLegion paths:")))
        print(indent(str(self.system.legion_paths), prefix="  "))

        print(bright(white("\nCommand:")))
        print(yellow(f"  {' '.join(quote(t) for t in self.cmd)}"))

        if log_env := sorted(self.custom_env_vars):
            print(bright(white("\nCustomized Environment:")))
            for k in log_env:
                v = self.env[k].rstrip()
                print(f"  {dim(green(k))}={yellow(v)}")

        print(cyan(f"\n{'-':-<80}"))

        print(flush=True)

    def _process_logging(self) -> None:
        # make sure we only run processing at most once on multi-rank
        if self.launcher.kind == "none" or self.launcher.rank_id != "0":
            return

        if self.config.profiling.profile:
            self._process_profiling()

        if self.config.debugging.dataflow or self.config.debugging.event:
            self._process_debugging()

    def _process_debugging(self) -> None:
        legion_spy_py = str(self.system.legion_paths.legion_spy_py)
        ranks = self.config.multi_node.ranks

        cmd: Command = (legion_spy_py,)

        dflag = "d" if self.config.debugging.dataflow else ""
        eflag = "e" if self.config.debugging.event else ""
        if dflag or eflag:
            cmd += ("-{dflag}{eflag}",)

        cmd += tuple(f"legate_{n}.log" for n in range(ranks))

        keep_logs = self._run_processing(cmd, "spy")

        # Clean Legion Spy files, unless the user is doing extra logging, in
        # which case their logs and Spy's logs will be in the same file.
        user_logging_levels = self.config.logging.user_logging_levels
        if user_logging_levels is None and not keep_logs:
            log_dir = self.config.logging.logdir
            ranks = self.config.multi_node.ranks
            for n in range(ranks):
                log_dir.joinpath(f"legate_{n}.log").unlink()

    def _process_profiling(self) -> None:
        legion_prof_py = str(self.system.legion_paths.legion_prof_py)
        ranks = self.config.multi_node.ranks

        cmd: Command = (legion_prof_py, "-o", "legate_prof")

        cmd += tuple(f"legate_{n}.prof" for n in range(ranks))

        keep_logs = self._run_processing(cmd, "profiler")

        if not keep_logs:
            log_dir = self.config.logging.logdir
            ranks = self.config.multi_node.ranks
            for n in range(ranks):
                log_dir.joinpath(f"legate_{n}.prof").unlink()

    def _run_processing(self, cmd: Command, tool: str) -> bool:
        cmdstr = " ".join(quote(t) for t in cmd)
        ranks = self.config.multi_node.ranks
        ranks_per_node = self.config.multi_node.ranks_per_node

        if ranks // ranks_per_node > 4:
            print(_TOOL_WARN.format(tool=tool, cmd=cmdstr), flush=True)
            keep_logs = True
        else:
            log_dir = self.config.logging.logdir
            if self.config.info.verbose:
                print(f"Running: {cmdstr}", flush=True)
            run(cmd, check=True, cwd=log_dir)

        return keep_logs
