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

from pytest_mock import MockerFixture

import legate.driver as m

# main function shadows main module
# def test___all__() -> None:

# The main() function is very simple, this test just confirms that
# all the expected plumbing is hooked up as it is supposed to be


def test_main(mocker: MockerFixture) -> None:
    import legate.driver.config
    import legate.driver.driver
    import legate.driver.system

    config_spy = mocker.spy(legate.driver.config.Config, "__init__")
    system_spy = mocker.spy(legate.driver.system.System, "__init__")
    driver_spy = mocker.spy(legate.driver.driver.Driver, "__init__")
    mocker.patch("legate.driver.driver.Driver.run", return_value=123)

    result = m.main(["foo", "bar"])

    assert config_spy.call_count == 1
    assert config_spy.call_args[0][1:] == (["foo", "bar"],)
    assert config_spy.call_args[1] == {}

    assert system_spy.call_count == 1
    assert system_spy.call_args[0][1:] == ()
    assert system_spy.call_args[1] == {}

    assert driver_spy.call_count == 1
    assert len(driver_spy.call_args[0]) == 3
    assert isinstance(driver_spy.call_args[0][1], legate.driver.config.Config)
    assert isinstance(driver_spy.call_args[0][2], legate.driver.system.System)
    assert driver_spy.call_args[1] == {}

    assert result == 123
