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
import platform
from functools import cached_property

from .util import LegatePaths, LegionPaths, get_legate_paths, get_legion_paths

__all__ = ("System",)


class System:
    """Encapsulate details of the current system, e.g. runtime paths and OS."""

    def __init__(self) -> None:
        self.env = dict(os.environ)

    @cached_property
    def legate_paths(self) -> LegatePaths:
        """All the current runtime Legate Paths

        Returns
        -------
            LegionPaths

        """
        return get_legate_paths()

    @cached_property
    def legion_paths(self) -> LegionPaths:
        """All the current runtime Legion Paths

        Returns
        -------
            LegionPaths

        """
        return get_legion_paths(self.legate_paths)

    @cached_property
    def os(self) -> str:
        """The OS for this system

        Raises
        ------
            RuntimeError, if OS is not supported

        Returns
        -------
            str

        """
        if (os := platform.system()) not in {"Linux", "Darwin"}:
            raise RuntimeError(f"Legate does not work on {os}")
        return os

    @cached_property
    def LIB_PATH(self) -> str:
        """An ld library path environment variable name suitable for the OS

        Returns
        -------
            str

        """
        return "LD_LIBRARY_PATH" if self.os == "Linux" else "DYLD_LIBRARY_PATH"
