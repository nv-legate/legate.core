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

from shlex import quote
from textwrap import indent
from typing import TYPE_CHECKING, Type, TypeVar

from ..utils.types import DataclassProtocol
from ..utils.ui import kvtable, rule, section, value

if TYPE_CHECKING:
    from ..utils.system import System
    from .driver import Driver

__all__ = (
    "object_to_dataclass",
    "print_verbose",
)


T = TypeVar("T", bound=DataclassProtocol)


def object_to_dataclass(obj: object, typ: Type[T]) -> T:
    """Automatically generate a dataclass from an object with appropriate
    attributes.

    Parameters
    ----------
    obj: object
        An object to pull values from (e.g. an argparse Namespace)

    typ:
        A dataclass type to generate from ``obj``

    Returns
    -------
        The generated dataclass instance

    """
    kws = {name: getattr(obj, name) for name in typ.__dataclass_fields__}
    return typ(**kws)


def print_verbose(
    system: System,
    driver: Driver | None = None,
) -> None:
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
        None

    """

    print(f"\n{rule('Legion Python Configuration')}")

    print(section("\nLegate paths:"))
    print(indent(str(system.legate_paths), prefix="  "))

    print(section("\nLegion paths:"))
    print(indent(str(system.legion_paths), prefix="  "))

    if driver:
        print(section("\nCommand:"))
        cmd = " ".join(quote(t) for t in driver.cmd)
        print(f"  {value(cmd)}")

        if keys := sorted(driver.custom_env_vars):
            print(section("\nCustomized Environment:"))
            print(
                indent(
                    kvtable(driver.env, delim="=", align=False, keys=keys),
                    prefix="  ",
                )
            )

    print(f"\n{rule()}")

    print(flush=True)
