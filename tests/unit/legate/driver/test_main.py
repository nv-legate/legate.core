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

import sys

import pytest
from pytest_mock import MockerFixture

import legate.driver as m

# main function shadows main module
# def test___all__() -> None:

# The main() function is very simple, this test just confirms that
# all the expected plumbing is hooked up as it is supposed to be


# TODO: this test with the fake argv path does not work for the way
# legate is installed in CI, so skip for now.
@pytest.mark.skip
def test_main(mocker: MockerFixture) -> None:
    import legate.driver.config
    import legate.driver.driver
    import legate.util.system

    config_spy = mocker.spy(legate.driver.config.Config, "__init__")
    system_spy = mocker.spy(legate.util.system.System, "__init__")
    driver_spy = mocker.spy(legate.driver.driver.LegateDriver, "__init__")
    mocker.patch("legate.driver.driver.LegateDriver.run", return_value=123)
    mocker.patch.object(sys, "argv", ["/some/path/foo", "bar"])

    result = m.main()

    assert config_spy.call_count == 1
    assert config_spy.call_args[0][1] == sys.argv
    assert config_spy.call_args[1] == {}

    assert system_spy.call_count == 1
    assert system_spy.call_args[0][1:] == ()
    assert system_spy.call_args[1] == {}

    assert driver_spy.call_count == 1
    assert len(driver_spy.call_args[0]) == 3
    assert isinstance(driver_spy.call_args[0][1], legate.driver.config.Config)
    assert isinstance(driver_spy.call_args[0][2], legate.util.system.System)
    assert driver_spy.call_args[1] == {}

    assert result == 123
