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
from typing import TYPE_CHECKING

from .. import install_info
from ..util.fs import read_c_define

if TYPE_CHECKING:
    from ..util.system import System
    from ..util.types import Command, EnvDict, LauncherType
    from .config import ConfigProtocol

__all__ = ("Launcher",)

RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)

LAUNCHER_VAR_PREFIXES = (
    "CONDA_",
    "LEGATE_",
    "LEGION_",
    "LG_",
    "REALM_",
    "GASNET_",
    "PYTHON",
    "UCX_",
    "NCCL_",
    "CUNUMERIC_",
    "NVIDIA_",
)


class Launcher:
    """A base class for custom launch handlers for Legate.

    Subclasses should set ``kind`` and ``cmd`` properties during their
    initialization.

    Parameters
    ----------
        config : Config

        system : System

    """

    kind: LauncherType

    cmd: Command

    # base class will attempt to set this
    detected_rank_id: str | None = None

    _config: ConfigProtocol

    _system: System

    _env: EnvDict | None = None

    _custom_env_vars: set[str] | None = None

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        self._config = config
        self._system = system

        if config.multi_node.ranks == 1:
            self.detected_rank_id = "0"
        else:
            for var in RANK_ENV_VARS:
                if var in system.env:
                    self.detected_rank_id = system.env[var]
                    break

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.kind == other.kind
            and self.cmd == other.cmd
            and self.env == other.env
        )

    @classmethod
    def create(cls, config: ConfigProtocol, system: System) -> Launcher:
        """Factory method for creating appropriate Launcher subclass based on
        user configuration.

        Parameters
        ----------
            config : Config

            system : System

        Returns
            Launcher

        """
        kind = config.multi_node.launcher
        if kind == "none":
            return SimpleLauncher(config, system)
        if kind == "mpirun":
            return MPILauncher(config, system)
        if kind == "jsrun":
            return JSRunLauncher(config, system)
        if kind == "srun":
            return SRunLauncher(config, system)

        raise RuntimeError(f"Unsupported launcher: {kind}")

    # Slightly annoying, but it is helpful for testing to avoid importing
    # legate unless necessary, so defined these two as properties since the
    # command env depends on legate/legion paths

    @property
    def env(self) -> EnvDict:
        """A system environment to use with this launcher process."""
        if self._env is None:
            self._env, self._custom_env_vars = self._compute_env()
        return self._env

    @property
    def custom_env_vars(self) -> set[str]:
        """The set of environment variables specificaly customized by us."""
        if self._custom_env_vars is None:
            self._env, self._custom_env_vars = self._compute_env()
        return self._custom_env_vars

    @staticmethod
    def is_launcher_var(name: str) -> bool:
        """Whether an environment variable name is relevant for the laucher."""
        return name.endswith("PATH") or any(
            name.startswith(prefix) for prefix in LAUNCHER_VAR_PREFIXES
        )

    def _compute_env(self) -> tuple[EnvDict, set[str]]:
        config = self._config
        system = self._system

        env = {}

        # We never want to save python byte code for legate
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # Set the path to the Legate module as an environment variable
        # The current directory should be added to PYTHONPATH as well
        extra_python_paths = []
        if "PYTHONPATH" in system.env:
            extra_python_paths.append(system.env["PYTHONPATH"])

        if system.legion_paths.legion_module is not None:
            extra_python_paths.append(str(system.legion_paths.legion_module))

        if system.legion_paths.legion_jupyter_module is not None:
            extra_python_paths.append(
                str(system.legion_paths.legion_jupyter_module)
            )

        env["PYTHONPATH"] = os.pathsep.join(extra_python_paths)

        # If using NCCL prefer parallel launch mode over cooperative groups,
        # as the former plays better with Realm.
        env["NCCL_LAUNCH_MODE"] = "PARALLEL"

        # Make sure GASNet initializes MPI with the right level of
        # threading support
        env["GASNET_MPI_THREAD"] = "MPI_THREAD_MULTIPLE"

        if config.multi_node.ranks > 1 and "ucx" in install_info.networks:
            # UCX-related environment variables
            env["UCX_CUDA_COPY_MAX_REG_RATIO"] = "1.0"
            env["UCX_RCACHE_PURGE_ON_FORK"] = "n"

            # Link to the UCX bootstrap plugin
            env["REALM_UCP_BOOTSTRAP_PLUGIN"] = str(
                system.legion_paths.legion_lib_path
                / "realm_ucp_bootstrap_mpi.so"
            )

        if config.core.gpus > 0:
            assert "LEGATE_NEED_CUDA" not in system.env
            env["LEGATE_NEED_CUDA"] = "1"

        if config.core.openmp > 0:
            assert "LEGATE_NEED_OPENMP" not in system.env
            env["LEGATE_NEED_OPENMP"] = "1"

        if config.multi_node.ranks > 1:
            assert "LEGATE_NEED_NETWORK" not in system.env
            env["LEGATE_NEED_NETWORK"] = "1"

        if config.info.progress:
            assert "LEGATE_SHOW_PROGRESS" not in system.env
            env["LEGATE_SHOW_PROGRESS"] = "1"

        if config.info.mem_usage:
            assert "LEGATE_SHOW_USAGE" not in system.env
            env["LEGATE_SHOW_USAGE"] = "1"

        # Configure certain limits
        LEGATE_MAX_DIM = system.env.get(
            "LEGATE_MAX_DIM",
            read_c_define(
                system.legion_paths.legion_defines_h, "LEGION_MAX_DIM"
            ),
        )
        if LEGATE_MAX_DIM is None:
            raise RuntimeError("Cannot determine LEGATE_MAX_DIM")
        env["LEGATE_MAX_DIM"] = LEGATE_MAX_DIM

        LEGATE_MAX_FIELDS = system.env.get(
            "LEGATE_MAX_FIELDS",
            read_c_define(
                system.legion_paths.legion_defines_h, "LEGION_MAX_FIELDS"
            ),
        )
        if LEGATE_MAX_FIELDS is None:
            raise RuntimeError("Cannot determine LEGATE_MAX_FIELDS")
        env["LEGATE_MAX_FIELDS"] = LEGATE_MAX_FIELDS

        assert env["LEGATE_MAX_DIM"] is not None
        assert env["LEGATE_MAX_FIELDS"] is not None

        # Special run modes
        if config.debugging.freeze_on_error:
            env["LEGION_FREEZE_ON_ERROR"] = "1"

        # Debugging options
        # TODO: consider also adding UCX_HANDLE_ERRORS=none if using ucx
        # which stops UCX from installing its own signal handler
        if system.env.get("PYTHONFAULTHANDLER", "") == "":
            env["REALM_BACKTRACE"] = "1"
        elif "REALM_BACKTRACE" in system.env:
            raise RuntimeError(
                "REALM_BACKTRACE and PYTHONFAULTHANDLER should not be both set"
            )

        if "CUTENSOR_LOG_LEVEL" not in system.env:
            env["CUTENSOR_LOG_LEVEL"] = "1"

        if config.debugging.gasnet_trace:
            env["GASNET_TRACEFILE"] = str(
                config.logging.logdir / "gasnet_%.log"
            )

        custom_env_vars = set(env)

        full_env = dict(system.env)
        full_env.update(env)

        return full_env, custom_env_vars


RANK_ERR_MSG = """\
Could not detect rank ID on multi-rank run with no --launcher provided. If you
want Legate to use a launcher, e.g. mpirun, internally (recommended), then you
need to specify which one to use by passing --launcher. Otherwise you need to
invoke the legate script itself through a launcher.
"""


class SimpleLauncher(Launcher):
    """A Launcher subclass for the "no launcher" case."""

    kind: LauncherType = "none"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        # bind.sh handles computing local and global rank id, even in the
        # simple case, just for consistency. But we do still check the known
        # rank env vars below in order to issue RANK_ERR_MSG if needed
        if config.multi_node.ranks > 1 and self.detected_rank_id is None:
            raise RuntimeError(RANK_ERR_MSG)

        self.cmd = ()


class MPILauncher(Launcher):
    """A Launcher subclass to use mpirun [1] for launching Legate processes.

    [1] https://www.open-mpi.org/doc/current/man1/mpirun.1.php

    """

    kind: LauncherType = "mpirun"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        ranks = config.multi_node.ranks
        ranks_per_node = config.multi_node.ranks_per_node

        cmd = ["mpirun", "-n", str(ranks)]

        cmd += ["--npernode", str(ranks_per_node)]
        cmd += ["--bind-to", "none"]
        cmd += ["--mca", "mpi_warn_on_fork", "0"]

        for var in self.env:
            if self.is_launcher_var(var):
                cmd += ["-x", var]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)


class JSRunLauncher(Launcher):
    """A Launcher subclass to use jsrun [1] for launching Legate processes.

    [1] https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=SSWRJV_10.1.0/jsm/jsrun.html  # noqa

    """

    kind: LauncherType = "jsrun"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        ranks = config.multi_node.ranks
        ranks_per_node = config.multi_node.ranks_per_node

        cmd = ["jsrun", "-n", str(ranks // ranks_per_node)]

        cmd += ["-r", "1"]
        cmd += ["-a", str(ranks_per_node)]
        cmd += ["-c", "ALL_CPUS"]
        cmd += ["-g", "ALL_GPUS"]
        cmd += ["-b", "none"]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)


class SRunLauncher(Launcher):
    """A Launcher subclass to use srun [1] for launching Legate processes.

    [1] https://slurm.schedmd.com/srun.html

    """

    kind: LauncherType = "srun"

    def __init__(self, config: ConfigProtocol, system: System) -> None:
        super().__init__(config, system)

        ranks = config.multi_node.ranks
        ranks_per_node = config.multi_node.ranks_per_node

        cmd = ["srun", "-n", str(ranks)]

        cmd += ["--ntasks-per-node", str(ranks_per_node)]

        if config.debugging.gdb or config.debugging.cuda_gdb:
            # Execute in pseudo-terminal mode when we need to be interactive
            cmd += ["--pty"]

        self.cmd = tuple(cmd + config.multi_node.launcher_extra)
