# Copyright 2023 NVIDIA Corporation
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

from .util.settings import PrioritizedSetting, Settings, convert_bool

__all__ = ("settings",)


class LegateRuntimeSettings(Settings):
    consensus: PrioritizedSetting[bool] = PrioritizedSetting(
        "consensus",
        "LEGATE_CONSENSUS",
        default=False,
        convert=convert_bool,
        help="""
        Whether to enable consensus match on single node (for testing).
        """,
    )

    cycle_check: PrioritizedSetting[bool] = PrioritizedSetting(
        "cycle_check",
        "LEGATE_CYCLE_CHECK",
        default=False,
        convert=convert_bool,
        help="""
        Whether to check for reference cycles involving RegionField objects on
        exit (developer option). When such cycles arise during execution they
        stop used RegionFields from being collected and reused for new Stores,
        thus increasing memory pressure. By default this check will miss any
        RegionField cycles the garbage collector collected during execution.

        Run gc.disable() at the beginning of the program to avoid this.
        """,
    )

    future_leak_check: PrioritizedSetting[bool] = PrioritizedSetting(
        "future_leak_check",
        "LEGATE_FUTURE_LEAK_CHECK",
        default=False,
        convert=convert_bool,
        help="""
        Whether to check for reference cycles keeping Future/FutureMap objects
        alive after Legate runtime exit (developer option). Such leaks can
        result in Legion runtime shutdown hangs.
        """,
    )


settings = LegateRuntimeSettings()
