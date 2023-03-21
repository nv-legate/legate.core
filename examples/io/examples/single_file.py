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
from legateio import IOArray, read_file, read_file_parallel

import legate.core as lg

runtime = lg.get_legate_runtime()

arr = np.arange(10)
c1 = IOArray.from_legate_data_interface(arr.__legate_data_interface__)
c1.to_file("test.dat")
runtime.issue_execution_fence()

c2 = read_file("test.dat", lg.int64)
assert np.array_equal(arr, np.asarray(c2))

c3 = read_file_parallel("test.dat", lg.int64, parallelism=2)
assert np.array_equal(arr, np.asarray(c3))

c4 = read_file_parallel("test.dat", lg.int64)
assert np.array_equal(arr, np.asarray(c4))
