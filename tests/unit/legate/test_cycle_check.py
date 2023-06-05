# Copyright 2022-2023 NVIDIA Corporation
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

import os
import subprocess
from pathlib import Path

import pytest

PROG_TEXT = """
import numpy as np
from legate.core import get_legate_runtime, types as ty
store = get_legate_runtime().core_context.create_store(
    ty.int32, shape=(4,), optimize_scalar=False
)
# initialize the RegionField backing the store
store.storage
# create a cycle
x = [store]
x.append(x)
"""


def test_cycle_check(tmp_path: Path) -> None:
    prog_file = tmp_path / "prog.py"
    prog_file.write_text(PROG_TEXT)
    env = os.environ.copy()
    env["LEGATE_CYCLE_CHECK"] = "1"
    output = subprocess.check_output(
        [
            "legate",
            prog_file,
            "--cpus",
            "1",
        ],
        env=env,
    )
    assert "found cycle!" in output.decode("utf-8")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
