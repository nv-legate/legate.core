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
import cunumeric as np  # type: ignore
import pytest
from collective import _get_legate_store, collective_test  # , user_lib


def test_no_collective() -> None:
    store = np.ones(shape=(100, 10), dtype=int)
    collective_test(
        _get_legate_store(store),
        (
            100,
            10,
        ),
        (
            10,
            10,
        ),
    )


def test_broadcast() -> None:
    store = np.ones(
        shape=(
            1,
            10,
        ),
        dtype=int,
    )
    collective_test(
        _get_legate_store(store),
        (
            100,
            10,
        ),
        (
            10,
            10,
        ),
    )


def test_partial_overlap() -> None:
    a = np.ones(shape=(100, 100))
    b = np.ones(shape=(100,))
    a.dot(b)
    # store = np.ones(shape = (100,10,), dtype = int)
    # collective_test(_get_legate_store(store), (100, 10,), (10,10,))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
