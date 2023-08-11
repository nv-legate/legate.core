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
"""Consolidate driver configuration from command-line and environment.

"""
from __future__ import annotations

import shlex
from argparse import Namespace
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Protocol

from ..util import colors
from ..util.types import (
    ArgList,
    DataclassMixin,
    LauncherType,
    object_to_dataclass,
)
from ..util.ui import warn
from .args import parser

__all__ = ("Config",)


@dataclass(frozen=True)
class MultiNode(DataclassMixin):
    nodes: int
    ranks_per_node: int
    not_control_replicable: bool
    launcher: LauncherType
    launcher_extra: list[str]

    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up launcher_extra to automaticaly handle quoted strings with
        # internal whitespace, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.launcher_extra:
            ex: list[str] = sum(
                (shlex.split(x) for x in self.launcher_extra), []
            )
            object.__setattr__(self, "launcher_extra", ex)

    @property
    def ranks(self) -> int:
        return self.nodes * self.ranks_per_node


@dataclass(frozen=True)
class Binding(DataclassMixin):
    cpu_bind: str | None
    mem_bind: str | None
    gpu_bind: str | None
    nic_bind: str | None


@dataclass(frozen=True)
class Core(DataclassMixin):
    cpus: int
    gpus: int
    openmp: int
    ompthreads: int
    utility: int


@dataclass(frozen=True)
class Memory(DataclassMixin):
    sysmem: int
    numamem: int
    fbmem: int
    zcmem: int
    regmem: int
    eager_alloc: int


@dataclass(frozen=True)
class Profiling(DataclassMixin):
    profile: bool
    cprofile: bool
    nvprof: bool
    nsys: bool
    nsys_targets: str  # TODO: multi-choice
    nsys_extra: list[str]

    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up nsys_extra to automaticaly handle quoted strings with
        # internal whitespace, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.nsys_extra:
            ex: list[str] = sum((shlex.split(x) for x in self.nsys_extra), [])
            object.__setattr__(self, "nsys_extra", ex)


@dataclass(frozen=True)
class Logging(DataclassMixin):
    def __post_init__(self, **kw: dict[str, Any]) -> None:
        # fix up logdir to be a real path, have to use __setattr__ for frozen
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if self.logdir:
            object.__setattr__(self, "logdir", Path(self.logdir))

    user_logging_levels: str | None
    logdir: Path
    log_to_file: bool


@dataclass(frozen=True)
class Debugging(DataclassMixin):
    gdb: bool
    cuda_gdb: bool
    memcheck: bool
    valgrind: bool
    freeze_on_error: bool
    gasnet_trace: bool
    dataflow: bool
    event: bool
    collective: bool
    spy_assert_warning: bool


@dataclass(frozen=True)
class Info(DataclassMixin):
    progress: bool
    mem_usage: bool
    verbose: bool
    bind_detail: bool


@dataclass(frozen=True)
class Other(DataclassMixin):
    timing: bool
    wrapper: list[str]
    wrapper_inner: list[str]
    module: str | None
    dry_run: bool
    rlwrap: bool


class ConfigProtocol(Protocol):
    _args: Namespace

    argv: ArgList

    user_script: Optional[str]
    user_opts: tuple[str, ...]
    multi_node: MultiNode
    binding: Binding
    core: Core
    memory: Memory
    profiling: Profiling
    logging: Logging
    debugging: Debugging
    info: Info
    other: Other


class Config:
    """A centralized configuration object that provides the information
    needed by the Legate driver in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList) -> None:
        self.argv = argv

        args = parser.parse_args(self.argv[1:])

        colors.ENABLED = args.color

        # only saving this for help with testing
        self._args = args

        self.user_script = args.command[0] if args.command else None
        self.user_opts = tuple(args.command[1:]) if self.user_script else ()

        # these may modify the args, so apply before dataclass conversions
        self._fixup_nocr(args)
        self._fixup_log_to_file(args)

        self.multi_node = object_to_dataclass(args, MultiNode)
        self.binding = object_to_dataclass(args, Binding)
        self.core = object_to_dataclass(args, Core)
        self.memory = object_to_dataclass(args, Memory)
        self.profiling = object_to_dataclass(args, Profiling)
        self.logging = object_to_dataclass(args, Logging)
        self.debugging = object_to_dataclass(args, Debugging)
        self.info = object_to_dataclass(args, Info)
        self.other = object_to_dataclass(args, Other)

    @cached_property
    def console(self) -> bool:
        """Whether we are starting Legate as an interactive console."""
        return self.user_script is None

    def _fixup_nocr(self, args: Namespace) -> None:
        # this is slightly duplicative of MultiNode.ranks property, but fixup
        # checks happen before sub-configs are initialized from args
        ranks = int(args.nodes) * int(args.ranks_per_node)

        if self.console and not args.not_control_replicable and ranks > 1:
            print(warn("Disabling control replication for interactive run"))
            args.not_control_replicable = True

    def _fixup_log_to_file(self, args: Namespace) -> None:
        # Spy output is dumped to the same place as other logging, so we must
        # redirect all logging to a file, even if the user didn't ask for it.
        if args.dataflow or args.event or args.collective:
            if args.user_logging_levels is not None and not args.log_to_file:
                print(
                    warn(
                        "Logging output is being redirected to a "
                        f"file in directory {args.logdir}"
                    )
                )
            args.log_to_file = True
