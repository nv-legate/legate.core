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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import legate.util.colors as colors
from legate.driver.config import (
    Binding,
    Core,
    Debugging,
    Info,
    Logging,
    Memory,
    MultiNode,
    Other,
    Profiling,
)
from legate.jupyter.args import parser
from legate.util.types import ArgList, DataclassMixin, object_to_dataclass

__all__ = ("Config",)


@dataclass(frozen=True)
class Kernel(DataclassMixin):
    user: bool
    prefix: str | None
    spec_name: str
    display_name: str


class Config:
    """A Jupyter-specific configuration object that provides the information
    needed by the Legate driver in order to run.

    Parameters
    ----------
    argv : ArgList
        command-line arguments to use when building the configuration

    """

    def __init__(self, argv: ArgList) -> None:
        self.argv = argv

        args = parser.parse_args(self.argv[1:])

        # only saving these for help with testing
        self._args = args

        colors.ENABLED = args.color

        if args.display_name is None:
            args.display_name = args.spec_name

        self.kernel = object_to_dataclass(args, Kernel)
        self.verbose = args.verbose

        # these are the values we leave configurable for the kernel
        self.multi_node = object_to_dataclass(args, MultiNode)
        self.core = object_to_dataclass(args, Core)
        self.memory = object_to_dataclass(args, Memory)

        # turn everything else off
        self.user_script: Optional[str] = None
        self.user_opts: tuple[str, ...] = ()
        self.binding = Binding(None, None, None, None)
        self.profiling = Profiling(False, False, False, False, "", [])
        self.logging = Logging(None, Path(), False)
        self.debugging = Debugging(
            False, False, False, False, False, False, False, False
        )
        self.info = Info(False, False, self.verbose > 0, False)
        self.other = Other(None, False, False)
