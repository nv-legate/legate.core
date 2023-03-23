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

import argparse

import cunumeric as np
from legateio import IOArray, read_even_tiles

import legate.core as lg


def test(
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    dataset_name: str,
    print_input: bool,
):
    if len(shape) != len(tile_shape):
        raise ValueError(
            f"Incompatible tile shape {tile_shape} for data shape {shape}"
        )

    print(f"Array shape: {shape}, tile shape: {tile_shape}")

    runtime = lg.get_legate_runtime()

    # Use cuNumeric to generate a random array to dump to a dataset
    arr = np.random.randint(low=1, high=9, size=shape).astype("int8")

    if print_input:
        print(arr)

    # Construct an IOArray from the cuNumeric ndarray
    c1 = IOArray.from_legate_data_interface(arr.__legate_data_interface__)

    # Dump the IOArray to a dataset of even tiles
    c1.to_even_tiles(dataset_name, tile_shape)

    runtime.issue_execution_fence(block=True)

    # Read the dataset into an IOArray
    c2 = read_even_tiles(dataset_name)

    # Convert the IOArray into a cuNumeric ndarray and perform a binary
    # operation, just to confirm in the profile that the partition from the
    # reader tasks is reused in the downstream tasks.
    c2 = np.asarray(c2) * 1
    assert np.array_equal(c2, arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs="+",
        default=(8, 8),
        dest="shape",
        help="Data shape",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        nargs="+",
        default=(3, 3),
        dest="tile_shape",
        help="Tile shape",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="test",
        dest="dataset",
        help="Dataset name",
    )
    parser.add_argument(
        "--print-input",
        default=False,
        dest="print_input",
        action="store_true",
        help="Print input",
    )
    args, _ = parser.parse_known_args()

    test(args.shape, args.tile_shape, args.dataset, args.print_input)
