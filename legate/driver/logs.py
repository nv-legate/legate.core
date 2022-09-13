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
"""

"""
from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from shlex import quote
from subprocess import run
from typing import Iterator

from .config import Config
from .launcher import Launcher
from .system import System
from .types import Command
from .ui import warn

__all__ = ("process_logs",)

_LOG_TOOL_WARN = """\
Skipping the processing of {tool} output, to avoid wasting
resources in a large allocation. Please manually run: {cmd}
"""


class LogHandler(metaclass=ABCMeta):
    config: Config
    system: System

    def __init__(self, config: Config, system: System) -> None:
        self.config = config
        self.system = system

    @abstractmethod
    def process(self) -> bool:
        ...

    @abstractmethod
    def cleanup(self, keep_logs: bool) -> None:
        ...

    def run_processing_cmd(self, cmd: Command, tool: str) -> bool:
        cmdstr = " ".join(quote(t) for t in cmd)
        ranks = self.config.multi_node.ranks
        ranks_per_node = self.config.multi_node.ranks_per_node
        keep_logs = self.config.logging.keep_logs

        if ranks // ranks_per_node > 4:
            print(
                warn(_LOG_TOOL_WARN.format(tool=tool, cmd=cmdstr)), flush=True
            )
            keep_logs = True
        else:
            log_dir = self.config.logging.logdir
            if self.config.info.verbose:
                print(f"Running: {cmdstr}", flush=True)
            run(cmd, check=True, cwd=log_dir)

        return keep_logs


class ProfilingHandler(LogHandler):
    def process(self) -> bool:
        legion_prof_py = str(self.system.legion_paths.legion_prof_py)
        ranks = self.config.multi_node.ranks

        cmd: Command = (legion_prof_py, "-o", "legate_prof")

        cmd += tuple(f"legate_{n}.prof" for n in range(ranks))

        return self.run_processing_cmd(cmd, "profiler")

    def cleanup(self, keep_logs: bool) -> None:
        if keep_logs:
            return

        log_dir = self.config.logging.logdir
        ranks = self.config.multi_node.ranks
        for n in range(ranks):
            log_dir.joinpath(f"legate_{n}.prof").unlink()


class DebuggingHandler(LogHandler):
    def process(self) -> bool:
        legion_spy_py = str(self.system.legion_paths.legion_spy_py)
        ranks = self.config.multi_node.ranks

        cmd: Command = (legion_spy_py,)

        dflag = "d" if self.config.debugging.dataflow else ""
        eflag = "e" if self.config.debugging.event else ""
        if dflag or eflag:
            cmd += ("-{dflag}{eflag}",)

        cmd += tuple(f"legate_{n}.log" for n in range(ranks))

        return self.run_processing_cmd(cmd, "spy")

    def cleanup(self, keep_logs: bool) -> None:
        # Clean Legion Spy files, unless the user is doing extra logging, in
        # which case their logs and Spy's logs will be in the same file.
        user_logging_levels = self.config.logging.user_logging_levels
        if user_logging_levels is None and not keep_logs:
            log_dir = self.config.logging.logdir
            ranks = self.config.multi_node.ranks
            for n in range(ranks):
                log_dir.joinpath(f"legate_{n}.log").unlink()


@contextmanager
def process_logs(
    config: Config, system: System, launcher: Launcher
) -> Iterator[tuple[LogHandler, ...]]:

    os.makedirs(config.logging.logdir, exist_ok=True)

    handlers: list[LogHandler] = []

    if launcher.kind != "none" or launcher.rank_id == "0":
        if config.profiling.profile:
            handlers.append(ProfilingHandler(config, system))

        if config.debugging.dataflow or config.debugging.event:
            handlers.append(DebuggingHandler(config, system))

    # yielding the handlers really just makes testing simpler
    yield tuple(handlers)

    for handler in handlers:
        handler.cleanup(handler.process())
