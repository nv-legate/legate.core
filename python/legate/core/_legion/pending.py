# Copyright 2021-2022 NVIDIA Corporation
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
from __future__ import annotations

from typing import Any

# We can't call out to the CFFI from inside of finalizer methods
# because that can risk a deadlock (CFFI's lock is stupid, they
# take it still in python so if a garbage collection is triggered
# while holding it you can end up deadlocking trying to do another
# CFFI call inside a finalizer because the lock is not reentrant).
# Therefore we defer deletions until we end up launching things
# later at which point we know that it is safe to issue deletions
_pending_unordered: dict[Any, Any] = dict()

# We also have some deletion operations which are only safe to
# be done if we know the Legion runtime is still running so we'll
# move them here and only issue the when we know we are inside
# of the execution of a task in some way
_pending_deletions: list[Any] = list()
