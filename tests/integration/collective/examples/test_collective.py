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
import pytest
from collective import (
    collective_test,
    collective_test_matvec,
    create_int64_store,
)

SHAPE = (
    100,
    10,
)
TILE = (
    10,
    10,
)


def test_no_collective() -> None:
    store = create_int64_store(shape=(100, 10))
    collective_test(store, SHAPE, TILE)


def test_broadcast() -> None:
    store = create_int64_store(shape=(1, 10))
    collective_test(store, SHAPE, TILE)


def test_overlap() -> None:
    a = create_int64_store(
        shape=(
            100,
            100,
        )
    )
    b = create_int64_store(shape=(100))
    c = create_int64_store(shape=(100))
    collective_test_matvec(a, b, c)


def test_transpose() -> None:
    a = create_int64_store(shape=(200,))

    a = a.slice(0, slice(None, 100))
    a = a.promote(0, 100)
    a = a.transpose(
        [
            1,
            0,
        ]
    )
    shape = (
        100,
        100,
    )

    collective_test(a, shape, TILE)


def test_project() -> None:
    shape = (
        100,
        100,
    )
    a = create_int64_store(shape)

    a = a.promote(0, 100)
    a = a.project(0, 1)
    a = a.promote(1, 100)
    a = a.project(1, 1)
    collective_test(a, shape, TILE)


def test_2_promotions() -> None:
    a = create_int64_store(shape=(1,))

    a = a.promote(0, 100)
    a = a.promote(0, 100)
    a = a.project(1, 1)
    shape = (
        100,
        100,
    )
    collective_test(a, shape, TILE)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
