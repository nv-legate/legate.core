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
from reduction import (
    matmul,
    multiply,
    print_store,
    sum_over_axis,
    to_cunumeric_array,
    user_context,
)

import legate.core.types as ty


def test(m: int, n: int, k: int, print_stores: bool, matmul_only: bool):
    # Generate inputs using cuNumeric
    rhs1 = user_context.create_store(ty.int64, (m, k))
    rhs2 = user_context.create_store(ty.int64, (k, n))
    tmp = np.arange(m * k).reshape(m, k)
    to_cunumeric_array(rhs1)[:] = tmp
    tmp = np.arange(k * n).reshape(k, n)
    to_cunumeric_array(rhs2)[:] = tmp

    if print_stores:
        print_store(rhs1)
        print_store(rhs2)

    if not matmul_only:
        # Implement matrix multiplication using sum_over_axis
        rhs1_promoted = rhs1.promote(2, n)
        rhs2_promoted = rhs2.promote(0, m)

        tmp = multiply(rhs1_promoted, rhs2_promoted)
        if print_stores:
            print_store(tmp)

        result = sum_over_axis(tmp, 1)
        if print_stores:
            print_store(result)

    result = matmul(rhs1, rhs2)
    if print_stores:
        print_store(result)

    # Bogus downstream computation to consume the matmul result
    to_cunumeric_array(result) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        type=int,
        default=3,
        dest="m",
        help="Extent of the first dimension",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        dest="n",
        help="Extent of the second dimension",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        dest="k",
        help="Extent of the contracting dimension",
    )
    parser.add_argument(
        "--print-stores",
        default=False,
        dest="print_stores",
        action="store_true",
        help="Print stores",
    )
    parser.add_argument(
        "--matmul-only",
        default=False,
        dest="matmul_only",
        action="store_true",
        help="Only call matmul",
    )
    args, _ = parser.parse_known_args()

    test(args.m, args.n, args.k, args.print_stores, args.matmul_only)
