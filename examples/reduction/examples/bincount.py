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
from reduction import bincount, user_context

import legate.core.types as ty


def test():
    size = 100
    num_bins = 10

    # Generate random inputs using cuNumeric
    store = user_context.create_store(ty.uint64, size)
    np.asarray(store)[:] = np.random.randint(
        low=0, high=num_bins - 1, size=size
    )
    print(np.asarray(store))

    result = bincount(store, num_bins)
    print(np.asarray(result))


if __name__ == "__main__":
    test()
