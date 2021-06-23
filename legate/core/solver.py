# Copyright 2021 NVIDIA Corporation
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

from .legion import Rect
from .partition import NoPartition


class Expression(object):
    def __repr__(self):
        return str(self)


class Offset(Expression):
    def __init__(self, expr, offset):
        self.expr = expr
        self.offset = offset

    @property
    def closed(self):
        return self.expr.closed

    def __str__(self):
        return f"{self.expr} + {self.offset}"

    def collapse(self):
        expr = self.expr.collapse()
        if isinstance(expr, Tile):
            return Tile(expr.tile_size, self.offset)
        else:
            return type(self)(expr, self.offset)

    def substitute(self, subst):
        return Offset(self.expr.substitute(subst), self.offset)

    def invert(self, rhs):
        assert not self.expr.closed
        return self.expr, Offset(rhs, -self.offset)

    def find_term(self):
        return self.expr.find_term()


class Tile(Expression):
    def __init__(self, tile_size, offset=0):
        self.tile_size = tile_size
        self.offset = offset

    @property
    def closed(self):
        return True

    def __str__(self):
        return f"tile({self.tile_size}, {self.offset})"

    def collapse(self):
        return self

    def substitute(self, subst):
        return self

    def invert(self, rhs):
        raise RuntimeError("Invalid inversion")

    def find_term(self):
        raise RuntimeError("Invalid call")


class Dimension(Expression):
    def __init__(self, index, shape):
        self.index = index
        self.shape = tuple(dim for dim in shape)

    @property
    def closed(self):
        return False

    def __eq__(self, other):
        if not isinstance(other, Expression):
            raise ValueError(f"Unknown expression type: {type(other)}")
        return Match(self, other)

    def __le__(self, other):
        if not isinstance(other, Expression):
            raise ValueError(f"Unknown expression type: {type(other)}")
        return Subsume(self, other)

    def __add__(self, other):
        if not isinstance(other, int):
            raise ValueError(f"Unknown offset type: {type(other)}")
        return Offset(self, other)

    def __str__(self):
        return f"{self.shape.name}_{self.index}"

    def collapse(self):
        raise RuntimeError("Dimension variable cannot be collapsed")

    def substitute(self, subst):
        return subst[self] if self in subst else self

    def invert(self, rhs):
        return self, rhs

    def find_term(self):
        return self

    def __hash__(self):
        return hash(repr(self))


class Constraint(object):
    def __init__(self, lhs, rhs, op):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __str__(self):
        return f"{self.lhs} {self.op} {self.rhs}"

    def __repr__(self):
        return str(self)

    @property
    def closed(self):
        return self.lhs.closed or self.rhs.closed

    def substitute(self, subst):
        lhs = self.lhs.substitute(subst)
        rhs = self.rhs.substitute(subst)
        return type(self)(lhs, rhs)


class Match(Constraint):
    def __init__(self, lhs, rhs):
        super(Match, self).__init__(lhs, rhs, "==")


class Subsume(Constraint):
    def __init__(self, lhs, rhs):
        super(Subsume, self).__init__(lhs, rhs, ">=")


class EqClass(object):
    def __init__(self):
        # Maps a store to the equivalent class id
        self._class_ids = {}
        self._next_class_id = 0
        # Maps an equivalent class id to the class
        self._classes = {}

    @property
    def empty(self):
        return self._next_class_id == 0

    def _add(self, store1, store2):
        cls = set([store1, store2])
        cls_id = self._next_class_id
        self._next_class_id + 1
        self._classes[cls_id] = cls
        self._class_ids[store1] = cls_id
        self._class_ids[store2] = cls_id

    def _update(self, store1, store2):
        cls_id = self._class_ids[store1]
        cls = self._classes[cls_id]
        cls.add(store2)
        self._class_ids[store2] = cls_id

    def _merge(self, store1, store2):
        cls_id1 = self._class_ids[store1]
        cls_id2 = self._class_ids[store2]
        cls = self._classes[cls_id1] | self._classes[cls_id2]
        self._classes[cls_id1] = cls
        self._classes[cls_id2] = cls

    def record(self, store1, store2):
        """
        Record an equivalence relation between two stores
        """
        found1 = store1 in self._class_ids
        found2 = store2 in self._class_ids

        if not found1 and not found2:
            self._add(store1, store2)
        elif found1:
            self._update(store1, store2)
        elif found2:
            self._update(store2, store1)
        else:
            self._merge(store1, store2)

    def copy(self):
        new = EqClass()
        new._class_ids = self._class_ids.copy()
        new._classes = self._classes.copy()
        return new

    def union(self, other):
        if self.empty:
            self._class_ids = other._class_ids.copy()
            self._classes = other._classes.copy()
        else:
            for other_class in other._classes.values():
                cls = other_class.copy()
                store1 = cls.pop()
                for store2 in cls:
                    self.record(store1, store2)

    def find(self, store):
        """
        Return an equivalence class for a given store.
        """
        if store not in self._class_ids:
            return set([store])
        else:
            return self._classes[self._class_ids[store]]


class Strategy(object):
    def __init__(self, launch_shape, strategy):
        self._launch_shape = launch_shape
        self._strategy = strategy

    def __getitem__(self, store):
        if store not in self._strategy:
            raise ValueError(f"No strategy is found for {store}")
        return self._strategy[store].get_requirement(store)

    def launch(self, launcher):
        if self._launch_shape is None:
            launcher.execute_single()
        else:
            launcher.execute(Rect(self._launch_shape))


class Partitioner(object):
    def __init__(self, runtime, ops):
        self._runtime = runtime
        self._ops = ops

    def partition_stores(self):
        stores = set()
        constraints = EqClass()
        for op in self._ops:
            stores.update(op.get_all_stores())
            constraints.union(op.constraints)

        partitions = {}
        prev_part = None
        while len(stores) > 0:
            store = stores.pop()
            if isinstance(prev_part, NoPartition):
                partition = prev_part
            else:
                partition = store.find_key_partition()

            cls = constraints.find(store)
            for to_align in cls:
                partitions[to_align] = partition
            stores = stores - cls
            prev_part = partition

        return Strategy(prev_part.color_shape, partitions)
