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

from collections.abc import MutableSet


class OrderedSet(MutableSet):
    """
    A set() variant whose iterator returns elements in insertion order.

    The implementation of this class piggybacks off of the corresponding
    iteration order guarantee for dict(), starting with Python 3.7. This is
    useful for guaranteeing symmetric execution of algorithms on different
    shards in a replicated context.
    """

    def __init__(self, copy_from=None):
        self._dict = {}
        if copy_from is not None:
            for obj in copy_from:
                self.add(obj)

    def add(self, obj):
        self._dict[obj] = None

    def update(self, other):
        for obj in other:
            self.add(obj)

    def discard(self, obj):
        self._dict.pop(obj, None)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, obj):
        return obj in self._dict

    def __iter__(self):
        return iter(self._dict)


def cast_tuple(value):
    return value if isinstance(value, tuple) else tuple(value)
