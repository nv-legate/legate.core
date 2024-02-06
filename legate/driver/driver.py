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

from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from shlex import quote
from subprocess import run
from textwrap import indent
from typing import TYPE_CHECKING, Any

from ..util.system import System
from ..util.types import DataclassMixin
from ..util.ui import kvtable, rule, section, value
from .command import CMD_PARTS_CANONICAL, CMD_PARTS_LEGION
from .config import ConfigProtocol
from .launcher import Launcher, SimpleLauncher

if TYPE_CHECKING:
    from ..util.types import Command, EnvDict

__all__ = ("LegateDriver", "CanonicalDriver", "format_verbose")


@dataclass(frozen=True)
class LegateVersions(DataclassMixin):
    """Collect package versions relevant to Legate."""

    legate_version: str


class LegateDriver:
    """Coordinate the system, user-configuration, and launcher to appropriately
    execute the Legate process.

    Parameters
    ----------
        config : Config

        system : System

    """

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        self.config = config
        self.system = system
        self.launcher = Launcher.create(config, system)

    @property
    def cmd(self) -> Command:
        """The full command invocation that should be used to start Legate."""
        config = self.config
        launcher = self.launcher
        system = self.system

        parts = (part(config, system, launcher) for part in CMD_PARTS_LEGION)
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

    @property
    def dry_run(self) -> bool:
        """Check verbose and dry run.

        Returns
        -------
            bool : whether dry run is enabled

        """
        if self.config.info.verbose:
            msg = format_verbose(self.system, self)
            self.print_on_head_node(msg, flush=True)

        return self.config.other.dry_run

    def run(self) -> int:
        """Run the Legate process.

        Returns
        -------
            int : process return code

        """
        if self.dry_run:
            return 0

        if self.config.other.timing:
            self.print_on_head_node(f"Legate start: {datetime.now()}")

        ret = run(self.cmd, env=self.env).returncode

        if self.config.other.timing:
            self.print_on_head_node(f"Legate end: {datetime.now()}")

        log_dir = self.config.logging.logdir

        if self.config.profiling.profile:
            self.print_on_head_node(
                f"Profiles have been generated under {log_dir}, run "
                f"legion_prof --view {log_dir}/legate_*.prof to view them"
            )

        if self.config.debugging.spy:
            self.print_on_head_node(
                f"Legion Spy logs have been generated under {log_dir}, run "
                f"legion_spy.py {log_dir}/legate_*.log to process them"
            )

        return ret

    def print_on_head_node(self, *args: Any, **kw: Any) -> None:
        launcher = self.launcher

        if launcher.kind != "none" or launcher.detected_rank_id == "0":
            print(*args, **kw)


class CanonicalDriver(LegateDriver):
    """Coordinate the system, user-configuration, and launcher to appropriately
    execute the Legate process.

    Parameters
    ----------
        config : Config

        system : System

    """

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        self.config = config
        self.system = system
        self.launcher = SimpleLauncher(config, system)

    @property
    def cmd(self) -> Command:
        """The full command invocation that should be used to start Legate."""
        config = self.config
        launcher = self.launcher
        system = self.system

        parts = (
            part(config, system, launcher) for part in CMD_PARTS_CANONICAL
        )
        return sum(parts, ())

    def run(self) -> int:
        """Run the Legate process.

        Returns
        -------
            int : process return code

        """
        assert False, "This function should not be invoked."


def get_versions() -> LegateVersions:
    from legate import __version__ as lg_version

    return LegateVersions(legate_version=lg_version)


def format_verbose(
    system: System,
    driver: LegateDriver | None = None,
) -> str:
    """Print system and driver configuration values.

    Parameters
    ----------
    system : System
        A System instance to obtain Legate and Legion paths from

    driver : Driver or None, optional
        If not None, a Driver instance to obtain command invocation and
        environment from (default: None)

    Returns
    -------
        str

    """
    out = StringIO()

    out.write(f"\n{rule('Legion Python Configuration')}\n")

    out.write(section("\nLegate paths:\n"))
    out.write(indent(str(system.legate_paths), prefix="  "))

    out.write(section("\n\nLegion paths:\n"))
    out.write(indent(str(system.legion_paths), prefix="  "))

    out.write(section("\n\nVersions:\n"))
    out.write(indent(str(get_versions()), prefix="  "))

    if driver:
        out.write(section("\n\nCommand:\n"))
        cmd = " ".join(quote(t) for t in driver.cmd)
        out.write(f"  {value(cmd)}")

        if keys := sorted(driver.custom_env_vars):
            out.write(section("\n\nCustomized Environment:\n"))
            out.write(
                indent(
                    kvtable(driver.env, delim="=", align=False, keys=keys),
                    prefix="  ",
                )
            )

    out.write(f"\n\n{rule()}\n")

    return out.getvalue()
