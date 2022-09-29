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

from subprocess import run

from .command import CMD_PARTS
from .config import Config
from .launcher import Launcher
from .logs import process_logs
from .system import System
from .types import Command, EnvDict
from .ui import warn
from .util import print_verbose

__all__ = ("Driver",)

_DARWIN_GDB_WARN = """\
You must start the debugging session with the following command,
as LLDB no longer forwards the environment to subprocesses for security
reasons:

(lldb) process launch -v LIB_PATH={libpath} -v PYTHONPATH={pythonpath}

"""


class Driver:
    """Coordinate the system, user-configuration, and launcher to appropriately
    execute the Legate process.

    Parameters
    ----------
        config : Config

        system : System

    """

    def __init__(self, config: Config, system: System) -> None:
        self.config = config
        self.system = system
        self.launcher = Launcher.create(config, system)

    @property
    def cmd(self) -> Command:
        """The full command invocation that should be used to start Legate."""
        config = self.config
        launcher = self.launcher
        system = self.system

        parts = (part(config, system, launcher) for part in CMD_PARTS)
        return launcher.cmd + sum(parts, ())

    @property
    def env(self) -> EnvDict:
        """The system environment that should be used when started Legate."""
        # in case we want to augment the launcher env we could do it here
        return self.launcher.env

    @property
    def custom_env_vars(self) -> set[str]:
        """The names of environment variables that we have explicitly set
        for the system environment.

        """
        # in case we want to augment the launcher env we could do it here
        return self.launcher.custom_env_vars

    def run(self) -> int:
        """Run the Legate process.

        Returns
        -------
            int : process return code

        """
        if self.config.info.verbose:
            print_verbose(self.system, self)

        self._darwin_gdb_warn()

        if self.config.other.dry_run:
            return 0

        with process_logs(self.config, self.system, self.launcher):
            return run(self.cmd, env=self.env).returncode

    def _darwin_gdb_warn(self) -> None:
        gdb = self.config.debugging.gdb

        if gdb and self.system.os == "Darwin":
            libpath = self.env[self.system.LIB_PATH]
            pythonpath = self.env["PYTHONPATH"]
            print(
                warn(
                    _DARWIN_GDB_WARN.format(
                        libpath=libpath, pythonpath=pythonpath
                    )
                )
            )
