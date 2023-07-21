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
from reduction import unique, user_context

import legate.core.types as ty


def test(n: int, radix: int, print_stores: bool):
    # Generate inputs using cuNumeric
    input = user_context.create_store(ty.int32, n)
    np.asarray(input)[:] = np.random.randint(
        low=0, high=10, size=n, dtype="int32"
    )

    if print_stores:
        print(np.asarray(input))

    result = unique(input, radix=radix)
    if print_stores:
        print(np.asarray(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        dest="n",
        help="Number of elements in the input store",
    )
    parser.add_argument(
        "-r",
        "--radix",
        type=int,
        default=4,
        dest="radix",
        help="Fan-in of the reduction tree",
    )
    parser.add_argument(
        "--print-stores",
        default=False,
        dest="print_stores",
        action="store_true",
        help="Print stores",
    )
    args, _ = parser.parse_known_args()

    test(args.n, args.radix, args.print_stores)
