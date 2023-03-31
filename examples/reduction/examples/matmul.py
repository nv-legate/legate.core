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


def test():
    m = 3
    k = 4
    n = 5

    rhs1 = user_context.create_store(ty.int64, (m, k))
    rhs2 = user_context.create_store(ty.int64, (k, n))
    tmp = np.arange(m * k).reshape(m, k)
    to_cunumeric_array(rhs1)[:] = tmp
    tmp = np.arange(k * n).reshape(k, n)
    to_cunumeric_array(rhs2)[:] = tmp
    print_store(rhs1)
    print_store(rhs2)

    rhs1_promoted = rhs1.promote(2, n)
    rhs2_promoted = rhs2.promote(0, m)

    tmp = multiply(rhs1_promoted, rhs2_promoted)
    print_store(tmp)

    result = sum_over_axis(tmp, 1)
    print_store(result)

    result = matmul(rhs1, rhs2)
    print_store(result)


if __name__ == "__main__":
    test()
