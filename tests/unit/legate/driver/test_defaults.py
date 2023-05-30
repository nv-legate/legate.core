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
from typing import Any

from pytest_mock import MockerFixture

import legate.driver.defaults as m


def test_LEGATE_CPUS() -> None:
    assert m.LEGATE_CPUS == 4


def test_LEGATE_GPUS() -> None:
    assert m.LEGATE_GPUS == 0


def test_LEGATE_NODES() -> None:
    assert m.LEGATE_NODES == 1


def test_LEGATE_RANKS_PER_NODE() -> None:
    assert m.LEGATE_RANKS_PER_NODE == 1


def test_LEGATE_EAGER_ALLOC_PERCENTAGE_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_EAGER_ALLOC_PERCENTAGE", "12345", m)
    assert m.LEGATE_EAGER_ALLOC_PERCENTAGE == 12345


def test_LEGATE_EAGER_ALLOC_PERCENTAGE_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_EAGER_ALLOC_PERCENTAGE", m)
    assert m.LEGATE_EAGER_ALLOC_PERCENTAGE == 50


def test_LEGATE_FBMEM_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_FBMEM", "12345", m)
    assert m.LEGATE_FBMEM == 12345


def test_LEGATE_FBMEM_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_FBMEM", m)
    assert m.LEGATE_FBMEM == 4000


def test_LEGATE_LOG_DIR(mocker: MockerFixture) -> None:
    mocker.patch("os.getcwd", return_value="foo")
    reload(m)
    assert m.LEGATE_LOG_DIR == "foo"


def test_LEGATE_NUMAMEM_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_NUMAMEM", "12345", m)
    assert m.LEGATE_NUMAMEM == 12345


def test_LEGATE_NUMAMEM_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_NUMAMEM", m)
    assert m.LEGATE_NUMAMEM == 0


def test_LEGATE_OMP_PROCS_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_OMP_PROCS", "12345", m)
    assert m.LEGATE_OMP_PROCS == 12345


def test_LEGATE_OMP_PROCS_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_OMP_PROCS", m)
    assert m.LEGATE_OMP_PROCS == 0


def test_LEGATE_OMP_THREADS_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_OMP_THREADS", "12345", m)
    assert m.LEGATE_OMP_THREADS == 12345


def test_LEGATE_OMP_THREADS_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_OMP_THREADS", m)
    assert m.LEGATE_OMP_THREADS == 4


def test_LEGATE_REGMEM_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_REGMEM", "12345", m)
    assert m.LEGATE_REGMEM == 12345


def test_LEGATE_REGMEM_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_REGMEM", m)
    assert m.LEGATE_REGMEM == 0


def test_LEGATE_SYSMEM_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_SYSMEM", "12345", m)
    assert m.LEGATE_SYSMEM == 12345


def test_LEGATE_SYSMEM_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_SYSMEM", m)
    assert m.LEGATE_SYSMEM == 4000


def test_LEGATE_UTILITY_CORES_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_UTILITY_CORES", "12345", m)
    assert m.LEGATE_UTILITY_CORES == 12345


def test_LEGATE_UTILITY_CORES_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_UTILITY_CORES", m)
    assert m.LEGATE_UTILITY_CORES == 2


def test_LEGATE_ZCMEM_env(set_and_reload: Any) -> None:
    set_and_reload("LEGATE_ZCMEM", "12345", m)
    assert m.LEGATE_ZCMEM == 12345


def test_LEGATE_ZCMEM_no_env(clear_and_reload: Any) -> None:
    clear_and_reload("LEGATE_ZCMEM", m)
    assert m.LEGATE_ZCMEM == 32
