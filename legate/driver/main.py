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
"""

"""
from __future__ import annotations

from typing import Type, Union

from . import CanonicalDriver, LegateDriver

__all__ = ("legate_main",)


def prepare_driver(
    argv: list[str],
    driver_type: Union[Type[CanonicalDriver], Type[LegateDriver]],
) -> Union[CanonicalDriver, LegateDriver]:
    from ..util.system import System
    from ..util.ui import error
    from . import Config
    from .driver import print_verbose

    try:
        config = Config(argv)
    except Exception as e:
        print(error("Could not configure driver:\n"))
        raise e

    try:
        system = System()
    except Exception as e:
        print(error("Could not determine System settings: \n"))
        raise e

    try:
        driver = driver_type(config, system)
    except Exception as e:
        msg = "Could not initialize driver, path config and exception follow:"  # noqa
        print(error(msg))
        print_verbose(system)
        raise e

    return driver


def legate_main(argv: list[str]) -> int:
    """A main function for the Legate driver that can be used programmatically
    or by entry-points.

    Parameters
    ----------
        argv : list[str]
            Command-line arguments to start the Legate driver with

    Returns
    -------
        int, a process return code

    """
    driver = prepare_driver(argv, LegateDriver)
    return driver.run()
