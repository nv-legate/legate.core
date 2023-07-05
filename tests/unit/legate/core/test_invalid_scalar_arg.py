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


class Test_scalar_arg:
    def test_unimplemented_types(self) -> None:
        context = get_legate_runtime().core_context
        # Create a task object only to test validation logic
        task = context.create_auto_task(0)
        with pytest.raises(NotImplementedError):
            task.add_scalar_arg(None, ty.struct_type([ty.int8]))
        with pytest.raises(NotImplementedError):
            task.add_scalar_arg(
                (1,), ty.array_type(ty.struct_type([ty.int8]), 1)
            )
        with pytest.raises(NotImplementedError):
            task.add_scalar_arg((1,), (ty.struct_type([ty.int8]),))

    def test_scalar_arg_with_array_type(self) -> None:
        context = get_legate_runtime().core_context
        # Create a task object only to test validation logic
        task = context.create_auto_task(0)
        with pytest.raises(ValueError):
            task.add_scalar_arg(1, ty.array_type(ty.int8, 1))

    def test_array_size_mismatch(self) -> None:
        context = get_legate_runtime().core_context
        # Create a task object only to test validation logic
        task = context.create_auto_task(0)
        with pytest.raises(ValueError):
            task.add_scalar_arg((1, 2, 3), ty.array_type(ty.int8, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
