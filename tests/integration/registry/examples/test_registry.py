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
from registry import user_context, user_lib


def test_task_registrar():
    task = user_context.create_auto_task(user_lib.shared_object.HELLO)
    task.execute()


def test_task_immediate():
    task = user_context.create_auto_task(user_lib.shared_object.WORLD)
    task.execute()


def test_task_invalid():
    with pytest.raises(ValueError):
        user_context.create_auto_task(12345)

    with pytest.raises(ValueError):
        user_context.create_auto_task(user_lib.shared_object.NO_VARIANT)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
