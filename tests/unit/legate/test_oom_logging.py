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

import os
import subprocess
from pathlib import Path

import pytest

PROG_TEXT = """
from legate.core import Annotation, ingest, float64, TiledSplit
import cunumeric as cn
import numpy as np

# test that ingested Stores show up
tile_shape = (4, 3)
colors = (2, 1)
shape = tuple(ci * di for (ci, di) in zip(colors, tile_shape))
def get_buffer(color):
    (a, b) = color
    return memoryview(np.ones(tile_shape))
tab = ingest( # 13: ingestion site of new array
    float64,
    shape,
    colors,
    TiledSplit(tile_shape),
    get_buffer,
)
arr = cn.array(tab) # 20: shouldn't show up; array used here but not created
arr += 1 # 21: shouldn't show up; array used here but not created

# test that unbound stores show up
x = cn.arange(13) # 24: regular array
y = cn.unique(x) # 25: unbound array

# not instantiated => not reported
z = cn.ones((1024 * 1024,)) # 28: shouldn't show up; never instantiated

# instantiated in subsequent add operation => reported
a = cn.ones((1024 * 1024,)) # 31: regular array

# will succeed
b = a + 1 # 34: regular array

# will succeed
c = a + 2 # 37: regular array

# will fail
d = a + 3 # 40: will be reported, as the failed allocation
"""

LINES_PRESENT = (13, 24, 25, 31, 34, 37, 40)

LINES_NOT_PRESENT = (20, 21, 28)


@pytest.mark.parametrize("full_bt", (False, True))
def test_oom_logging(tmp_path: Path, full_bt: bool) -> None:
    prog_file = tmp_path / "prog.py"
    prog_file.write_text(PROG_TEXT)
    env = os.environ.copy()
    if full_bt:
        env["LEGATE_FULL_BT_ON_OOM"] = "1"
    env["LEGATE_TEST"] = "1"
    output = subprocess.run(
        [
            "legate",
            "--cpus",
            "1",
            "--sysmem",
            "50",
            prog_file,
        ],
        env=env,
        capture_output=True,
    )
    assert output.returncode != 0
    stdout = output.stdout.decode("utf-8")

    print(stdout)
    assert "Out of memory" in stdout
    if not full_bt:
        assert "create_store" not in stdout
    for line in LINES_PRESENT:
        assert f"prog.py:{line}" in stdout
        if full_bt:
            assert f'prog.py", line {line}, in <module>' in stdout
    for line in LINES_NOT_PRESENT:
        assert f"prog.py:{line}" not in stdout
        if full_bt:
            assert f'prog.py", line {line}, in <module>' not in stdout


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
