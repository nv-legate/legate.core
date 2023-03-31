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

from reduction import (
    print_store,
    sum_over_axis,
    to_cunumeric_array,
    user_context,
)

import legate.core.types as ty


def test():
    store = user_context.create_store(ty.int64, (4, 5))
    to_cunumeric_array(store).fill(1)
    print_store(store)

    result1 = sum_over_axis(store, 0)
    print_store(result1)

    result2 = sum_over_axis(store, 1)
    print_store(result2)


if __name__ == "__main__":
    test()
