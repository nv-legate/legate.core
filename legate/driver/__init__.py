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

from .config import Config
from .driver import LegateDriver, CanonicalDriver
from .launcher import Launcher


def main() -> int:
    import os, shlex, sys

    from .main import legate_main as _main

    # A little explanation. We want to encourage configuration options be
    # passed via LEGATE_CONFIG, in order to be considerate to user scripts.
    # But we still need to accept actual command line args for comaptibility,
    # and those should also take precedences. Here we splice the options from
    # LEGATE_CONFIG in before sys.argv, and take advantage of the fact that if
    # there are any options repeated in both places, argparse will use the
    # latter (i.e. the actual command line provided ones).
    env_args = shlex.split(os.environ.get("LEGATE_CONFIG", ""))
    argv = sys.argv[:1] + env_args + sys.argv[1:]

    return _main(argv)
