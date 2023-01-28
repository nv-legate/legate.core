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

from .util.args import ArgSpec, Argument

ARGS = [
    Argument(
        "consensus",
        ArgSpec(
            action="store_true",
            default=False,
            dest="consensus",
            help="Turn on consensus match on single node (for testing).",
        ),
    ),
    Argument(
        "cycle-check",
        ArgSpec(
            action="store_true",
            default=False,
            dest="cycle_check",
            help=(
                "Check for reference cycles involving RegionField objects on "
                "script exit (developer option). When such cycles arise "
                "during execution, they stop used RegionFields from getting "
                "collected and reused for new Stores, thus increasing memory "
                "pressure. By default this check will miss any RegionField "
                "cycles the garbage collector collected during execution; "
                "run gc.disable() at the beginning of the program to avoid "
                "this."
            ),
        ),
    ),
    Argument(
        "future-leak-check",
        ArgSpec(
            action="store_true",
            default=False,
            dest="future_leak_check",
            help=(
                "Check for reference cycles keeping Future/FutureMap objects "
                "alive after Legate runtime exit (developer option). Such "
                "leaks can result in Legion runtime shutdown hangs."
            ),
        ),
    ),
]
