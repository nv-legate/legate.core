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

from importlib import reload
from types import ModuleType
from typing import Any, Callable, Iterable

import pytest

from legate.driver import Config, Launcher
from legate.driver.config import MultiNode
from legate.util.system import System

from .util import GenConfig, GenSystem


@pytest.fixture
def clear_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, ModuleType], None]:
    def _inner(name_or_names: str | Iterable[str], m: ModuleType) -> None:
        if isinstance(name_or_names, str):
            name_or_names = [name_or_names]

        for name in name_or_names:
            monkeypatch.delenv(name, raising=False)

        reload(m)

    return _inner


@pytest.fixture
def set_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, str, ModuleType], None]:
    def _inner(name: str, value: str, m: ModuleType) -> None:
        monkeypatch.setenv(name, value)
        reload(m)

    return _inner


@pytest.fixture
def genconfig() -> Any:
    def _config(
        args: list[str] | None = None,
        *,
        fake_module: str | None = "foo.py",
        multi_rank: tuple[int, int] | None = None,
    ) -> Config:
        args = ["legate"] + (args or [])
        if fake_module:
            args += [fake_module]

        config = Config(args)

        if multi_rank:
            # This is annoying but we can only replace the entire dataclass
            nocr = config.multi_node.not_control_replicable
            launcher = config.multi_node.launcher
            launcher_extra = config.multi_node.launcher_extra
            config.multi_node = MultiNode(
                *multi_rank, nocr, launcher, launcher_extra
            )

        return config

    return _config


@pytest.fixture
def gensystem(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _system(
        rank_env: dict[str, str] | None = None, os: str | None = None
    ) -> System:
        if rank_env:
            for k, v in rank_env.items():
                monkeypatch.setenv(k, v)
        system = System()
        if os:
            monkeypatch.setattr(system, "os", os)
        return system

    return _system


@pytest.fixture
def genobjs(
    genconfig: GenConfig, gensystem: GenSystem, monkeypatch: pytest.MonkeyPatch
) -> Any:
    def _objs(
        args: list[str] | None = None,
        *,
        fake_module: str | None = "foo.py",
        multi_rank: tuple[int, int] | None = None,
        rank_env: dict[str, str] | None = None,
        os: str | None = None,
    ) -> tuple[Config, System, Launcher]:
        config = genconfig(
            args, fake_module=fake_module, multi_rank=multi_rank
        )
        system = gensystem(rank_env, os)
        launcher = Launcher.create(config, system)
        return config, system, launcher

    return _objs
