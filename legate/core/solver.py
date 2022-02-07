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

from .constraints import Alignment, Broadcast, Containment
from .legion import Future, Rect
from .partition import REPLICATE, Replicate
from .shape import Shape
from .utils import OrderedSet


class EqClass(object):
    def __init__(self):
        # Maps a variable to the equivalent class id
        self._class_ids = {}
        self._next_class_id = 0
        # Maps an equivalent class id to the class
        self._classes = {}

    @property
    def empty(self):
        return self._next_class_id == 0

    def _add(self, var1, var2):
        cls = set([var1, var2])
        cls_id = self._next_class_id
        self._next_class_id + 1
        self._classes[cls_id] = cls
        self._class_ids[var1] = cls_id
        self._class_ids[var2] = cls_id

    def _update(self, var1, var2):
        cls_id = self._class_ids[var1]
        cls = self._classes[cls_id]
        cls.add(var2)
        self._class_ids[var2] = cls_id

    def _merge(self, var1, var2):
        cls_id1 = self._class_ids[var1]
        cls_id2 = self._class_ids[var2]
        cls = self._classes[cls_id1] | self._classes[cls_id2]
        self._classes[cls_id1] = cls
        self._classes[cls_id2] = cls

    def record(self, var1, var2):
        """
        Record an equivalence relation between two vars
        """
        found1 = var1 in self._class_ids
        found2 = var2 in self._class_ids

        if not found1 and not found2:
            self._add(var1, var2)
        elif found1:
            self._update(var1, var2)
        elif found2:
            self._update(var2, var1)
        else:
            self._merge(var1, var2)

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
                var1 = cls.pop()
                for var2 in cls:
                    self.record(var1, var2)

    def find(self, var):
        """
        Return an equivalence class for a given var.
        """
        if var not in self._class_ids:
            return set([var])
        else:
            return self._classes[self._class_ids[var]]


class Strategy(object):
    def __init__(self, launch_shape, strategy, fspaces, key_parts):
        if launch_shape is not None:
            self._launch_domain = Rect(hi=launch_shape)
        else:
            self._launch_domain = None
        self._strategy = strategy
        self._fspaces = fspaces
        self._key_parts = key_parts

    @property
    def parallel(self):
        return self._launch_domain is not None

    @property
    def launch_domain(self):
        return self._launch_domain

    @property
    def launch_ndim(self):
        if self._launch_domain is None:
            return 1
        else:
            return self._launch_domain.dim

    def set_launch_domain(self, launch_domain):
        if self._launch_domain is not None:
            raise RuntimeError(
                "Manual task launch cannot be used when some stores are "
                "auto-partitioned"
            )
        self._launch_domain = launch_domain

    def get_projection(self, part):
        partition = self.get_partition(part)
        return partition.get_requirement(self.launch_ndim, part.store)

    def get_partition(self, part):
        assert not part.store.unbound
        if part not in self._strategy:
            raise ValueError(f"No strategy is found for {part}")
        return self._strategy[part]

    def get_field_space(self, part):
        assert part.store.unbound
        if part not in self._fspaces:
            raise ValueError(f"No strategy is found for {part}")
        return self._fspaces[part]

    def is_key_part(self, part):
        return part in self._key_parts

    def launch(self, launcher):
        if self.parallel:
            return launcher.execute(self.launch_domain)
        else:
            return launcher.execute_single()

    def __str__(self):
        st = "[Strategy]"
        for part, partition in self._strategy.items():
            st += f"\n{part} ~~> {partition}"
        for part, fspace in self._fspaces.items():
            st += f"\n{part} ~~> {fspace}"
        return st

    def __repr__(self):
        return str(self)


class Partitioner(object):
    def __init__(self, runtime, ops, must_be_single=False):
        self._runtime = runtime
        self._ops = ops
        self._must_be_single = must_be_single

    def _solve_broadcast_constraints(
        self, unknowns, constraints, broadcasts, partitions
    ):
        to_remove = OrderedSet()
        for unknown in unknowns:
            store = unknown.store
            if not (store.kind is Future or unknown in broadcasts):
                continue

            to_remove.add(unknown)

            if unknown in partitions:
                continue

            if store.kind is Future:
                partitions[unknown] = REPLICATE
            else:
                cls = constraints.find(unknown)
                for to_align in cls:
                    partitions[to_align] = REPLICATE

        return unknowns - to_remove

    def _solve_unbound_constraints(
        self, unknowns, constraints, partitions, fspaces
    ):
        to_remove = OrderedSet()
        for unknown in unknowns:
            store = unknown.store
            if not store.unbound:
                continue

            to_remove.add(unknown)

            if unknown in partitions:
                continue

            cls = constraints.find(unknown)
            assert all(to_align.store.unbound for to_align in cls)

            fspace = self._runtime.create_field_space()
            for to_align in cls:
                partitions[unknown] = REPLICATE
                fspaces[unknown] = fspace

        return unknowns - to_remove, len(to_remove) > 0

    @staticmethod
    def _find_restrictions(cls):
        merged = None
        for unknown in cls:
            store = unknown.store
            restrictions = store.find_restrictions()
            if merged is None:
                merged = restrictions
            else:
                merged = tuple(min(a, b) for a, b in zip(merged, restrictions))
        return merged

    def _find_all_restrictions(self, unknowns, constraints):
        all_restrictions = {}
        for unknown in unknowns:
            if unknown in all_restrictions:
                continue
            cls = constraints.find(unknown)
            restrictions = self._find_restrictions(cls)
            for store in cls:
                all_restrictions[unknown] = restrictions
        return all_restrictions

    def partition_stores(self):
        unknowns = OrderedSet()
        constraints = EqClass()
        broadcasts = OrderedSet()
        dependent = {}
        for op in self._ops:
            unknowns.update(op.all_unknowns)
            for c in op.constraints:
                if isinstance(c, Alignment):
                    constraints.record(c._lhs, c._rhs)
                elif isinstance(c, Broadcast):
                    broadcasts.add(c._expr)
                elif isinstance(c, Containment):
                    if c._rhs in dependent:
                        raise NotImplementedError(
                            "Partitions constrained by multiple constraints "
                            "are not supported yet"
                        )
                    dependent[c._rhs] = c._lhs

        if self._must_be_single or len(unknowns) == 0:
            broadcasts = unknowns

        partitions = {}
        fspaces = {}

        unknowns = self._solve_broadcast_constraints(
            unknowns,
            constraints,
            broadcasts,
            partitions,
        )

        unknowns, must_be_1d_launch = self._solve_unbound_constraints(
            unknowns,
            constraints,
            partitions,
            fspaces,
        )

        all_restrictions = self._find_all_restrictions(unknowns, constraints)

        def cost(unknown):
            store = unknown.store
            return (
                -store.comm_volume(),
                not store.has_key_partition(all_restrictions[unknown]),
            )

        unknowns = sorted(unknowns, key=cost)

        key_parts = set()
        prev_part = None
        for unknown in unknowns:
            if unknown in partitions:
                continue
            elif unknown in dependent:
                continue

            store = unknown.store
            restrictions = all_restrictions[unknown]

            if isinstance(prev_part, Replicate):
                partition = prev_part
            else:
                partition = store.compute_key_partition(restrictions)
                key_parts.add(unknown)

            cls = constraints.find(unknown)
            for to_align in cls:
                if to_align in partitions:
                    continue
                partitions[to_align] = partition

            prev_part = partition

        for lhs, rhs in dependent.items():
            rhs = rhs.subst(partitions).reduce()
            partitions[lhs] = rhs._part

        color_shape = None if prev_part is None else prev_part.color_shape

        if must_be_1d_launch and color_shape is not None:
            color_shape = Shape((color_shape.volume(),))

        return Strategy(color_shape, partitions, fspaces, key_parts)
