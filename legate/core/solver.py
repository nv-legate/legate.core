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

from . import Future, Rect
from .constraints import Alignment, Broadcast, Containment
from .partition import REPLICATE
from .shape import Shape
from .utils import OrderedSet


def join_restrictions(x, y):
    return tuple(min(a, b) for a, b in zip(x, y))


class EqClass:
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


class Strategy:
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

    def __str__(self):
        st = "[Strategy]"
        st += f"\nLaunch domain: {self._launch_domain}"
        for part, partition in self._strategy.items():
            st += f"\n{part} ~~> {partition}"
        for part, fspace in self._fspaces.items():
            st += f"\n{part} ~~> {fspace}"
        return st

    def __repr__(self):
        return str(self)


class Partitioner:
    def __init__(self, runtime, ops, must_be_single=False):
        self._runtime = runtime
        self._ops = ops
        self._must_be_single = must_be_single

    def _solve_constraints_for_futures(
        self, unknowns, constraints, partitions
    ):
        to_remove = OrderedSet()
        for unknown in unknowns:
            store = unknown.store
            if store.kind is not Future:
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

        unbound_ndims = set(unknown.store.ndim for unknown in to_remove)

        if len(unbound_ndims) > 1:
            raise NotImplementedError(
                "Unbound stores for an operation must have the same "
                "number of dimensions for now"
            )

        unbound_ndim = unbound_ndims.pop() if len(unbound_ndims) == 1 else None
        return unknowns - to_remove, unbound_ndim

    @staticmethod
    def _find_restrictions(cls, broadcasts):
        merged = None
        for unknown in cls:
            store = unknown.store
            restrictions = store.find_restrictions()
            if unknown in broadcasts:
                restrictions = join_restrictions(
                    broadcasts[unknown], restrictions
                )
            if merged is None:
                merged = restrictions
            else:
                merged = join_restrictions(merged, restrictions)
        return merged

    def _find_all_restrictions(self, unknowns, broadcasts, constraints):
        all_restrictions = {}
        for unknown in unknowns:
            if unknown in all_restrictions:
                continue
            cls = constraints.find(unknown)
            restrictions = self._find_restrictions(cls, broadcasts)
            for store in cls:
                all_restrictions[unknown] = restrictions
        return all_restrictions

    def maybe_find_alternative_key_partition(
        self, chosen_partition, original, cls, restrictions, must_be_even
    ):
        original_comm_vol = original.store.comm_volume()
        # If there is a store in the equivalence class that has an even
        # key partition, we use it instead
        for unknown in cls:
            store = unknown.store
            # Careful! these are partition symbols and we overrode the equal
            # operator on them to mean alignments, so we compare between
            # their identities.
            if original is unknown:
                continue
            elif not store.has_key_partition(restrictions):
                continue
            # We don't want to use the store's key partition if that store
            # incurs a bigger amount of data movement
            elif store.comm_volume() < original_comm_vol:
                continue

            part = store.compute_key_partition(restrictions)
            if part.even:
                return part, unknown

        # TODO: For now we repartition the store used as a center of a stencil
        # if it was previously partitioned unevenly. If we have a proper
        # affine dependent partitioning calls, we can avoid this.
        if original in must_be_even:
            store = original.store
            store.reset_key_partition()
            return store.compute_key_partition(restrictions), original
        else:
            return chosen_partition, original

    @staticmethod
    def compute_launch_shape(partitions, all_outputs, unbound_ndim):
        # We filter out the cases where any of the outputs is assigned
        # to replication, in which case the operation must be performed
        # sequentially
        for unknown, part in partitions.items():
            if unknown.store in all_outputs and part is REPLICATE:
                return None

        # If we're here, this means that replicated stores are safe to access
        # in parallel, so we filter those out to determine the launch domain
        parts = [part for part in partitions.values() if part is not REPLICATE]

        # If all stores are replicated, we can't parallelize the operation
        if len(parts) == 0:
            return None

        # Here we check if all partitions agree on the color shape
        must_be_1d_launch = False
        launch_shape = parts[0].color_shape
        for part in parts[1:]:
            if part.color_shape != launch_shape:
                # When some partitions have different color shapes,
                # a 1D launch space is the only option
                must_be_1d_launch = True
                break

        if must_be_1d_launch:
            # If this operation has a multi-dimensional unbound store,
            # we can't use a 1-D launch domain, hence falling back to
            # a sequential launch
            if unbound_ndim != 1:
                return None

            # If all color spaces don't have the same number of colors,
            # it means some inputs are much smaller than the others
            # to be partitioned into the same number of pieces.
            # We simply serialize the launch in that case for now.
            volumes = set(part.color_shape.volume() for part in parts)
            if len(volumes) > 1:
                return None
            else:
                return Shape(volumes)
        # If there is an unbound store, the store's dimensionality must be
        # the same as that of the launch domain
        elif unbound_ndim is None or unbound_ndim == launch_shape.ndim:
            return launch_shape
        else:
            return None

    def partition_stores(self):
        unknowns = OrderedSet()
        constraints = EqClass()
        broadcasts = {}
        dependent = {}
        must_be_even = OrderedSet()
        all_outputs = set()
        for op in self._ops:
            unknowns.update(op.all_unknowns)
            for c in op.constraints:
                if isinstance(c, Alignment):
                    constraints.record(c._lhs, c._rhs)
                elif isinstance(c, Broadcast):
                    broadcasts[c._expr] = c._restrictions
                elif isinstance(c, Containment):
                    if c._rhs in dependent:
                        raise NotImplementedError(
                            "Partitions constrained by multiple constraints "
                            "are not supported yet"
                        )
                    for unknown in c._lhs.unknowns():
                        must_be_even.add(unknown)
                    dependent[c._rhs] = c._lhs
        for op in self._ops:
            all_outputs.update(
                store for store in op.outputs if not store.unbound
            )

        if self._must_be_single or len(unknowns) == 0:
            for unknown in unknowns:
                c = unknown.broadcast()
                broadcasts[unknown] = c._restrictions

        partitions = {}
        fspaces = {}

        unknowns = self._solve_constraints_for_futures(
            unknowns,
            constraints,
            partitions,
        )

        unknowns, unbound_ndim = self._solve_unbound_constraints(
            unknowns,
            constraints,
            partitions,
            fspaces,
        )

        all_restrictions = self._find_all_restrictions(
            unknowns, broadcasts, constraints
        )

        def cost(unknown):
            store = unknown.store
            return (
                -store.comm_volume(),
                not store.has_key_partition(all_restrictions[unknown]),
            )

        unknowns = sorted(unknowns, key=cost)

        key_parts = set()
        for unknown in unknowns:
            if unknown in partitions:
                continue
            elif unknown in dependent:
                continue

            store = unknown.store
            restrictions = all_restrictions[unknown]
            cls = constraints.find(unknown)

            partition = store.compute_key_partition(restrictions)
            if not partition.even and len(cls) > 1:
                partition, unknown = self.maybe_find_alternative_key_partition(
                    partition,
                    unknown,
                    cls,
                    restrictions,
                    must_be_even,
                )
            key_parts.add(unknown)

            for to_align in cls:
                if to_align in partitions:
                    continue
                partitions[to_align] = partition

        for rhs, lhs in dependent.items():
            lhs = lhs.subst(partitions).reduce()
            partitions[rhs] = lhs._part

        launch_shape = self.compute_launch_shape(
            partitions, all_outputs, unbound_ndim
        )

        return Strategy(launch_shape, partitions, fspaces, key_parts)
