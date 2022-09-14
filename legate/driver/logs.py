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

__all__ = (
    "DebuggingHandler",
    "LogHandler",
    "process_logs",
    "ProfilingHandler",
)

_LOG_TOOL_WARN = """\
Skipping the processing of {tool} output, to avoid wasting
resources in a large allocation. Please manually run: {cmd}
"""


class LogHandler(metaclass=ABCMeta):
    """A base class for handling log output from external tools such as
    debuggers or profilers that can be run along with Legate.

    Subclasses must implement ``process`` and ``cleanup`` methods.

    """

    config: Config
    system: System

    def __init__(self, config: Config, system: System) -> None:
        self.config = config
        self.system = system

    @abstractmethod
    def process(self) -> bool:
        """Perform processing of log files from external tools."""
        ...

    @abstractmethod
    def cleanup(self, keep_logs: bool) -> None:
        """Clean up and remove log files left by external tools"""
        ...

    def run_processing_cmd(self, cmd: Command, tool: str) -> bool:
        """A helper for running log-processing commands, as long as the
        allocation is not too large.

        Returns a boolean indicating whether log files should be kept or not.
        The value from the user config may be overridden, e.g. if processing
        is skipped because of allocation size.

        Parameters
        ----------
            cmd : Command
                A command invocation for ``subprocess.run``

            tool : str
                The name of the external tool, for display purposes

        Returns
        -------
            bool : whether to keep log files or clean them up

        """
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
    """A LogHandler subclass for .prof log files."""

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
    """A LogHandler subclass for legion_spy .log files."""

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
    """A context manager for log initializion and processing/cleanup, based
    on the user configuration.

    Paramerters
        config : Config

        system : System

        launcher : Launcher

    Returns
        tuple[LogHandler] : All the handlers created, based on user config

    """

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
