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
from region_manager import user_context, user_lib

import legate.core.types as ty


def test_region_manager():
    task = user_context.create_auto_task(user_lib.cffi.TESTER)
    for _ in range(2000):
        store = user_context.create_store(ty.int64, shape=(10,))
        task.add_output(store)
    task.execute()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
