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

LEGION_WARNING = """

All Legate programs must be run with a legion_python interperter. We
recommend that you use the Legate driver script "bin/legate" found
in the installation directory to launch Legate programs as it
provides easy-to-use flags for invoking legion_python. You can see
options for using the driver script with "bin/legate --help". You
can also invoke legion_python directly.

Use "bin/legate --verbose ..." to see some examples of how to call
legion_python directly.
"""

# TODO (bv) temp transitive imports until cunumeric is updated
from .util.args import (  # noqa
    ArgSpec,
    Argument,
    parse_library_command_args as parse_command_args,
)


def has_legion_context() -> bool:
    """Determine whether we are running in legion_python.

    Returns
        bool : True if running in legion_python, otherwise False

    """
    try:
        from legion_cffi import lib

        return lib.legion_runtime_has_context()
    except (ModuleNotFoundError, AttributeError):
        return False


def check_legion(msg: str = LEGION_WARNING) -> None:
    """Raise an error if we are not running in legion_python."""
    pass
    #if not has_legion_context():
    #    raise RuntimeError(msg)
