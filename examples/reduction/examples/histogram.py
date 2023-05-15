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
from reduction import bincount, categorize, histogram, user_context

import legate.core.types as ty


def test(size: int, num_bins: int):
    input = user_context.create_store(ty.int32, size)
    np.asarray(input)[:] = np.random.randint(
        low=0, high=size - 1, size=size, dtype="int32"
    )
    bins = user_context.create_store(ty.int32, (num_bins + 1,))
    np.asarray(bins)[:] = np.array(
        [size * v // num_bins for v in range(num_bins + 1)]
    )
    print("Input:")
    print(np.asarray(input))
    print("Bin edges:")
    print(np.asarray(bins))

    tmp = categorize(input, bins)
    result = bincount(tmp, num_bins)
    print("Histogram via bincount:")
    print(np.asarray(result))

    result = histogram(input, bins)
    print("Direct histogram implementation:")
    print(np.asarray(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        type=int,
        default=10,
        dest="size",
        help="Number of elements",
    )
    parser.add_argument(
        "-b",
        type=int,
        default=3,
        dest="num_bins",
        help="Number of bins",
    )
    args, _ = parser.parse_known_args()

    test(args.size, args.num_bins)
