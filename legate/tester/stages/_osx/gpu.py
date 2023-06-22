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
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..test_stage import TestStage
from ..util import UNPIN_ENV, Shard

if TYPE_CHECKING:
    from ....util.types import ArgList, EnvDict
    from ... import FeatureType
    from ...config import Config
    from ...test_system import TestSystem


class GPU(TestStage):
    """A test stage for exercising GPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: TestSystem
        Process execution wrapper

    """

    kind: FeatureType = "cuda"

    args: ArgList = []

    def __init__(self, config: Config, system: TestSystem) -> None:
        raise RuntimeError("GPU test are not supported on OSX")

    def env(self, config: Config, system: TestSystem) -> EnvDict:
        return dict(UNPIN_ENV)

    def delay(self, shard: Shard, config: Config, system: TestSystem) -> None:
        time.sleep(config.gpu_delay / 1000)
