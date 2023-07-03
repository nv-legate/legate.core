# Copyright 2022 NVIDIA Corporation
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

from legate.core import get_legate_runtime, types as ty


class Test_type_cache:
    def test_cache(self) -> None:
        runtime = get_legate_runtime()
        t_i64_2_1 = runtime.find_or_create_array_type(ty.int64, 2)
        t_i64_2_2 = runtime.find_or_create_array_type(ty.int64, 2)
        t_i64_3 = runtime.find_or_create_array_type(ty.int64, 3)
        t_i32_2 = runtime.find_or_create_array_type(ty.int32, 2)

        assert t_i64_2_1 is t_i64_2_2
        assert t_i64_2_1.uid != t_i64_3.uid
        assert t_i64_2_1.uid != t_i32_2.uid


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
