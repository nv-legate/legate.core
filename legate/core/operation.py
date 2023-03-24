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

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import legate.core.types as ty

from . import Future, FutureMap, Rect
from .constraints import PartSym
from .launcher import CopyLauncher, FillLauncher, TaskLauncher
from .partition import REPLICATE, Weighted
from .shape import Shape
from .store import Store, StorePartition
from .utils import OrderedSet, capture_traceback_repr

if TYPE_CHECKING:
    from .communicator import Communicator
    from .constraints import Constraint
    from .context import Context
    from .launcher import Proj
    from .projection import ProjFn, ProjOut, SymbolicPoint
    from .solver import Strategy
    from .types import DTType


class OperationProtocol(Protocol):
    _context: Context
    _mapper_id: int
    _op_id: int
    _inputs: list[Store]
    _outputs: list[Store]
    _reductions: list[tuple[Store, int]]
    _unbound_outputs: list[int]
    _scalar_outputs: list[int]
    _scalar_reductions: list[int]
    _partitions: dict[Store, list[PartSym]]
    _constraints: list[Constraint]
    _all_parts: list[PartSym]
    _launch_domain: Union[Rect, None]
    _error_on_interference: bool

    def launch(self, strategy: Strategy) -> None:
        ...

    @property
    def context(self) -> Context:
        return self._context

    @property
    def mapper_id(self) -> int:
        return self._mapper_id

    @property
    def can_raise_exception(self) -> bool:
        return False

    def capture_traceback(self) -> None:
        raise TypeError("Generic operation doesn't support capture_tracback")

    @property
    def inputs(self) -> list[Store]:
        return self._inputs

    @property
    def outputs(self) -> list[Store]:
        return self._outputs

    @property
    def reductions(self) -> list[tuple[Store, int]]:
        return self._reductions

    @property
    def unbound_outputs(self) -> list[int]:
        return self._unbound_outputs

    @property
    def scalar_outputs(self) -> list[int]:
        return self._scalar_outputs

    @property
    def scalar_reductions(self) -> list[int]:
        return self._scalar_reductions

    @property
    def constraints(self) -> list[Constraint]:
        return self._constraints

    @property
    def all_unknowns(self) -> list[PartSym]:
        return self._all_parts


class TaskProtocol(OperationProtocol, Protocol):
    _task_id: int
    _scalar_args: list[tuple[Any, Union[DTType, tuple[DTType]]]]
    _comm_args: list[Communicator]


class Operation(OperationProtocol):
    def __init__(
        self,
        context: Context,
        mapper_id: int,
        op_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._context = context
        self._mapper_id = mapper_id
        self._op_id = op_id
        self._inputs: list[Store] = []
        self._outputs: list[Store] = []
        self._reductions: list[tuple[Store, int]] = []
        self._unbound_outputs: list[int] = []
        self._scalar_outputs: list[int] = []
        self._scalar_reductions: list[int] = []
        self._partitions: dict[Store, list[PartSym]] = {}
        self._constraints: list[Constraint] = []
        self._all_parts: list[PartSym] = []
        self._launch_domain: Union[Rect, None] = None
        self._error_on_interference = True
        self._provenance = (
            None
            if context.provenance is None
            else (f"{context.provenance}$" f"{context.get_all_annotations()}")
        )

    @property
    def provenance(self) -> Optional[str]:
        return self._provenance

    def get_all_stores(self) -> OrderedSet[Store]:
        result: OrderedSet[Store] = OrderedSet()
        result.update(self._inputs)
        result.update(self._outputs)
        result.update(store for (store, _) in self._reductions)
        return result

    def add_alignment(self, store1: Store, store2: Store) -> None:
        """
        Sets an alignment between stores. Equivalent to the following code:

        ::

            symb1 = op.declare_partition(store1)
            symb2 = op.declare_partition(store2)
            op.add_constraint(symb1 == symb2)

        Parameters
        ----------
        store1, store2 : Store
            Stores to align

        Raises
        ------
        ValueError
            If the stores don't have the same shape or only one of them is
            unbound
        """
        self._check_store(store1, allow_unbound=True)
        self._check_store(store2, allow_unbound=True)
        if not (
            (store1.unbound and store2.unbound)
            or (store1.shape == store2.shape)
        ):
            raise ValueError(
                "Stores must have the same shape to be aligned, "
                f"but got {store1.shape} and {store2.shape}"
            )
        part1 = self._get_unique_partition(store1)
        part2 = self._get_unique_partition(store2)
        self.add_constraint(part1 == part2)

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        """
        Sets a broadcasting constraint on the store. Equivalent to the
        following code:

        ::

            symb = op.declare_partition(store)
            op.add_constraint(symb.broadcast(axes))

        Parameters
        ----------
        store : Store
            Store to set a broadcasting constraint on
        axes : int or Iterable[int], optional
            Axes to broadcast. The entire store is replicated if no axes are
            given.
        """
        self._check_store(store)
        part = self._get_unique_partition(store)
        self.add_constraint(part.broadcast(axes=axes))

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Adds a partitioning constraint to the operation

        Parameters
        ----------
        constraint : Constraint
            Partitioning constraint
        """
        self._constraints.append(constraint)

    def execute(self) -> None:
        """
        Submits the operation to the runtime. There is no guarantee that the
        operation will start the execution right upon the return of this
        method.
        """
        self._context.runtime.submit(self)

    @staticmethod
    def _check_store(store: Store, allow_unbound: bool = False) -> None:
        if not isinstance(store, Store):
            raise ValueError(f"Expected a Store, but got {type(store)}")
        elif not allow_unbound and store.unbound:
            raise ValueError("Expected a bound Store")

    def _get_unique_partition(self, store: Store) -> PartSym:
        if store not in self._partitions:
            return self.declare_partition(store)

        parts = self._partitions[store]
        if len(parts) > 1:
            raise RuntimeError(
                "Ambiguous store argument. When multple partitions exist for "
                "this store, a partition should be specified."
            )
        return parts[0]

    def get_tag(self, strategy: Strategy, part: PartSym) -> int:
        if strategy.is_key_part(part):
            return 1  # LEGATE_CORE_KEY_STORE_TAG
        else:
            return 0

    def _get_symbol_id(self) -> int:
        return len(self._all_parts)

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Operation(uid:{self._op_id})"

    def declare_partition(
        self, store: Store, disjoint: bool = True, complete: bool = True
    ) -> PartSym:
        """
        Creates a partition symbol for the store

        Parameters
        ----------
        store : Store
            Store to associate the partition symbol with
        disjoint : bool, optional
            ``True`` (by default) means the partition must be disjoint
        complete : bool, optional
            ``True`` (by default) means the partition must be complete

        Returns
        -------
        PartSym
            A partition symbol
        """
        sym = PartSym(
            self._op_id,
            self.get_name(),
            store,
            self._get_symbol_id(),
            disjoint=disjoint,
            complete=complete,
        )
        if store not in self._partitions:
            self._partitions[store] = [sym]
        else:
            self._partitions[store].append(sym)
        self._all_parts.append(sym)
        return sym


class Task(TaskProtocol):
    def __init__(
        self,
        task_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._task_id = task_id
        self._scalar_args: list[tuple[Any, Union[DTType, tuple[DTType]]]] = []
        self._comm_args: list[Communicator] = []
        self._exn_types: list[type] = []
        self._tb_repr: Union[None, str] = None
        self._side_effect = False
        self._concurrent = False

    @property
    def side_effect(self) -> bool:
        """
        Indicates whether the task has side effects

        Returns
        -------
        bool
            ``True`` if the task has side efects
        """
        return self._side_effect

    def set_side_effect(self, side_effect: bool) -> None:
        """
        Sets whether the task has side effects or not. A task is assumed to be
        free of side effects by default if the task only has scalar arguments.

        Parameters
        ----------
        side_effect : bool
            A new boolean value indicating whether the task has side effects
        """
        self._side_effect = side_effect

    @property
    def concurrent(self) -> bool:
        """
        Indicates whether the task needs a concurrent task launch.

        A concurrent task launch guarantees that all tasks will be active at
        the same time and make progress concurrently. This means that the tasks
        will and should be mapped to distinct processors and that no other
        tasks will be interleaved at any given point in time during execution
        of the concurrent tasks. This operational guarantee is useful
        when the tasks need to perform collective operations or explicit
        communication outside Legate, but comes with performance overhead
        due to distributed rendezvous used in the launch.

        Returns
        -------
        bool
            ``True`` if the task needs a concurrent task launch
        """
        return self._concurrent

    def set_concurrent(self, concurrent: bool) -> None:
        """
        Sets whether the task needs a concurrent task launch. Any task with at
        least one communicator will implicitly use concurrent task launch, so
        this method is to be used when the task needs a concurrent task launch
        for a reason unknown to Legate.

        Parameters
        ----------
        concurrent : bool
            A new boolean value indicating whether the task needs a concurrent
            task launch
        """
        self._concurrent = concurrent

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Task(tid:{self._task_id}, uid:{self._op_id})"

    def add_scalar_arg(
        self, value: Any, dtype: Union[DTType, tuple[DTType]]
    ) -> None:
        """
        Adds a by-value argument to the task

        Parameters
        ----------
        value : Any
            Scalar value or a tuple of scalars (but no nested tuples)
        dtype : DType
            Data type descriptor for the scalar value. A descriptor ``(T,)``
            means that the value is a tuple of elements of type ``T``.
        """

        self._scalar_args.append((value, dtype))

    def add_dtype_arg(self, dtype: DTType) -> None:
        code = self._context.type_system[dtype].code
        self._scalar_args.append((code, ty.int32))

    def throws_exception(self, exn_type: type) -> None:
        """
        Declares that the task can raise an exception. If more than one
        exception is added to the task, they are numbered by the order in which
        they are added, and those numbers are used to refer to them in the C++
        task.

        Parameters
        ----------
        exn_type : Type
            Type of exception
        """
        self._exn_types.append(exn_type)

    @property
    def can_raise_exception(self) -> bool:
        """
        Indicates whether the task can raise an exception

        Returns
        -------
        bool
            ``True`` if the task can raise an exception
        """
        return len(self._exn_types) > 0

    def capture_traceback(self) -> None:
        self._tb_repr = capture_traceback_repr()

    def _add_scalar_args_to_launcher(self, launcher: TaskLauncher) -> None:
        for arg, dtype in self._scalar_args:
            launcher.add_scalar_arg(arg, dtype)

    def _demux_scalar_stores_future(self, result: Future) -> None:
        num_unbound_outs = len(self.unbound_outputs)
        num_scalar_outs = len(self.scalar_outputs)
        num_scalar_reds = len(self.scalar_reductions)
        runtime = self.context.runtime

        num_all_scalars = (
            num_unbound_outs
            + num_scalar_outs
            + num_scalar_reds
            + int(self.can_raise_exception)
        )

        if num_all_scalars == 0:
            return
        elif num_all_scalars == 1:
            if num_scalar_outs == 1:
                output = self.outputs[self.scalar_outputs[0]]
                output.set_storage(result)
            elif num_scalar_reds == 1:
                (output, _) = self.reductions[self.scalar_reductions[0]]
                output.set_storage(result)
            elif self.can_raise_exception:
                runtime.record_pending_exception(
                    self._exn_types, result, self._tb_repr
                )
            else:
                assert num_unbound_outs == 1
        else:
            idx = len(self.unbound_outputs)
            for out_idx in self.scalar_outputs:
                output = self.outputs[out_idx]
                output.set_storage(runtime.extract_scalar(result, idx))
                idx += 1
            for red_idx in self.scalar_reductions:
                (output, _) = self.reductions[red_idx]
                output.set_storage(runtime.extract_scalar(result, idx))
                idx += 1
            if self.can_raise_exception:
                runtime.record_pending_exception(
                    self._exn_types,
                    runtime.extract_scalar(result, idx),
                    self._tb_repr,
                )

    def _demux_scalar_stores_future_map(
        self,
        result: FutureMap,
        launch_domain: Rect,
    ) -> None:
        num_unbound_outs = len(self.unbound_outputs)
        num_scalar_outs = len(self.scalar_outputs)
        num_scalar_reds = len(self.scalar_reductions)
        runtime = self.context.runtime

        num_all_scalars = (
            num_unbound_outs
            + num_scalar_outs
            + num_scalar_reds
            + int(self.can_raise_exception)
        )
        launch_shape = Shape(c + 1 for c in launch_domain.hi)
        assert num_scalar_outs == 0

        if num_all_scalars == 0:
            return
        elif num_all_scalars == 1:
            if num_scalar_reds == 1:
                (output, redop) = self.reductions[self.scalar_reductions[0]]
                redop_id = output.type.reduction_op_id(redop)
                output.set_storage(runtime.reduce_future_map(result, redop_id))
            elif num_unbound_outs == 1:
                output = self.outputs[self.unbound_outputs[0]]
                # TODO: need to track partitions for N-D unbound stores
                if output.ndim == 1:
                    partition = Weighted(launch_shape, result)
                    output.set_key_partition(partition)
            elif self.can_raise_exception:
                runtime.record_pending_exception(
                    self._exn_types,
                    runtime.reduce_exception_future_map(result),
                    self._tb_repr,
                )
            else:
                assert False
        else:
            idx = 0
            # TODO: We can potentially deduplicate these extraction tasks
            # by grouping output stores that are mapped to the same field space
            for out_idx in self.unbound_outputs:
                output = self.outputs[out_idx]
                # TODO: need to track partitions for N-D unbound stores
                if output.ndim > 1:
                    continue
                weights = runtime.extract_scalar_with_domain(
                    result, idx, launch_domain
                )
                partition = Weighted(launch_shape, weights)
                output.set_key_partition(partition)
                idx += 1
            for red_idx in self.scalar_reductions:
                (output, redop) = self.reductions[red_idx]
                redop_id = output.type.reduction_op_id(redop)
                data = runtime.extract_scalar_with_domain(
                    result, idx, launch_domain
                )
                output.set_storage(runtime.reduce_future_map(data, redop_id))
                idx += 1
            if self.can_raise_exception:
                exn_fut_map = runtime.extract_scalar_with_domain(
                    result, idx, launch_domain
                )
                runtime.record_pending_exception(
                    self._exn_types,
                    runtime.reduce_exception_future_map(exn_fut_map),
                    self._tb_repr,
                )

    def _demux_scalar_stores(
        self,
        result: Union[Future, FutureMap],
        launch_domain: Union[Rect, None],
    ) -> None:
        if launch_domain is None:
            assert isinstance(result, Future)
            self._demux_scalar_stores_future(result)
        else:
            assert isinstance(result, FutureMap)
            if launch_domain.get_volume() == 1:
                future = result.get_future(launch_domain.lo)
                self._demux_scalar_stores_future(future)
            else:
                self._demux_scalar_stores_future_map(result, launch_domain)

    def add_nccl_communicator(self) -> None:
        """
        Adds a NCCL communicator to the task
        """
        comm = self._context.get_nccl_communicator()
        self._comm_args.append(comm)

    def add_cpu_communicator(self) -> None:
        """
        Adds a CPU communicator to the task
        """
        comm = self._context.get_cpu_communicator()
        self._comm_args.append(comm)

    def _add_communicators(
        self, launcher: TaskLauncher, launch_domain: Union[Rect, None]
    ) -> None:
        if launch_domain is None:
            return
        for comm in self._comm_args:
            handle = comm.get_handle(launch_domain)
            launcher.add_communicator(handle)
        if any(comm.needs_barrier for comm in self._comm_args):
            launcher.insert_barrier()


class AutoOperation(Operation):
    def __init__(
        self,
        context: Context,
        mapper_id: int,
        op_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            context=context, mapper_id=mapper_id, op_id=op_id, **kwargs
        )

        self._input_parts: list[PartSym] = []
        self._output_parts: list[PartSym] = []
        self._reduction_parts: list[PartSym] = []

    def get_requirement(
        self, store: Store, part_symb: PartSym, strategy: Strategy
    ) -> tuple[Proj, int, StorePartition]:
        store_part = store.partition(strategy.get_partition(part_symb))
        req = store_part.get_requirement(strategy.launch_ndim)
        tag = self.get_tag(strategy, part_symb)
        return req, tag, store_part


class AutoTask(AutoOperation, Task):
    """
    A type of tasks that are automatically parallelized
    """

    def __init__(
        self,
        context: Context,
        task_id: int,
        mapper_id: int,
        op_id: int,
    ) -> None:
        super().__init__(
            context=context,
            task_id=task_id,
            mapper_id=mapper_id,
            op_id=op_id,
        )
        self._reusable_stores: list[Tuple[Store, PartSym]] = []
        self._reuse_map: dict[int, Store] = {}

    def add_input(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        """
        Adds a store as input to the task

        Parameters
        ----------
        store : Store
            Store to pass as input
        partition : PartSym, optional
            Partition to associate with the store. The default partition is
            picked if none is given.
        """
        self._check_store(store)
        if partition is None:
            partition = self._get_unique_partition(store)
        self._inputs.append(store)
        self._input_parts.append(partition)

    def add_output(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        """
        Adds a store as output to the task

        Parameters
        ----------
        store : Store
            Store to pass as output
        partition : PartSym, optional
            Partition to associate with the store. The default partition is
            picked if none is given.
        """
        self._check_store(store, allow_unbound=True)
        if store.kind is Future:
            self._scalar_outputs.append(len(self._outputs))
        elif store.unbound:
            self._unbound_outputs.append(len(self._outputs))
        if partition is None:
            partition = self._get_unique_partition(store)
        self._outputs.append(store)
        self._output_parts.append(partition)

    def add_reduction(
        self, store: Store, redop: int, partition: Optional[PartSym] = None
    ) -> None:
        """
        Adds a store to the task for reduction

        Parameters
        ----------
        store : Store
            Store to pass for reduction
        redop : int
            Reduction operator ID
        partition : PartSym, optional
            Partition to associate with the store. The default partition is
            picked if none is given.
        """
        self._check_store(store)
        if store.kind is Future:
            self._scalar_reductions.append(len(self._reductions))
        if partition is None:
            partition = self._get_unique_partition(store)
        self._reductions.append((store, redop))
        self._reduction_parts.append(partition)

    def record_reuse(
        self,
        strategy: Strategy,
        out_idx: int,
        store: Store,
        part_symb: PartSym,
    ) -> None:
        found = -1
        for idx, pair in enumerate(self._reusable_stores):
            to_reuse, to_reuse_part_symb = pair
            if store.type.size != to_reuse.type.size:
                continue
            if store.extents != to_reuse.extents:
                continue
            if not strategy.aligned(to_reuse_part_symb, part_symb):
                continue
            found = idx
            break

        if found != -1:
            to_reuse, to_reuse_part_symb = self._reusable_stores[found]
            self._reuse_map[out_idx] = to_reuse
            strategy.unify_key_part(part_symb, to_reuse_part_symb)
            self._reusable_stores = (
                self._reusable_stores[:found]
                + self._reusable_stores[found + 1 :]
            )

    def find_all_reusable_store_pairs(self, strategy: Strategy) -> None:
        # We attempt to reuse a store's storage only when the following
        # conditions are satisfied:
        #
        # 1. the source is marked linear and has no transforms
        # 2. the target is initialized by this operation (i.e., does not have
        #    a storage yet) and also has no transforms
        # 3. the source and target have the same storage size
        # 4. the source and target are aligned

        self._reusable_stores.extend(
            pair
            for pair in zip(self._inputs, self._input_parts)
            if pair[0].linear and not pair[0].transformed
        )
        if len(self._reusable_stores) == 0:
            return
        for idx, (store, part_symb) in enumerate(
            zip(self._outputs, self._output_parts)
        ):
            if store.unbound or store.kind is Future:
                continue
            if store._storage.has_data or store.transformed:
                continue
            self.record_reuse(strategy, idx, store, part_symb)

    def launch(self, strategy: Strategy) -> None:
        launcher = TaskLauncher(
            self.context,
            self._task_id,
            self.mapper_id,
            side_effect=self._side_effect,
            provenance=self.provenance,
        )

        self.find_all_reusable_store_pairs(strategy)

        for store, part_symb in zip(self._inputs, self._input_parts):
            req, tag, _ = self.get_requirement(store, part_symb, strategy)
            launcher.add_input(store, req, tag=tag)

        for idx, (store, part_symb) in enumerate(
            zip(self._outputs, self._output_parts)
        ):
            if store.unbound:
                continue
            if idx in self._reuse_map:
                store.move_data(self._reuse_map[idx])
            req, tag, store_part = self.get_requirement(
                store, part_symb, strategy
            )
            launcher.add_output(store, req, tag=tag)
            # We update the key partition of a store only when it gets updated
            store.set_key_partition(store_part.partition)

        for (store, redop), part_symb in zip(
            self._reductions, self._reduction_parts
        ):
            req, tag, store_part = self.get_requirement(
                store, part_symb, strategy
            )

            can_read_write = store_part.is_disjoint_for(strategy.launch_domain)
            req.redop = store.type.reduction_op_id(redop)

            launcher.add_reduction(
                store, req, tag=tag, read_write=can_read_write
            )

        for store, part_symb in zip(self._outputs, self._output_parts):
            if not store.unbound:
                continue
            fspace = strategy.get_field_space(part_symb)
            field_id = fspace.allocate_field(store.type)
            launcher.add_unbound_output(store, fspace, field_id)

        self._add_scalar_args_to_launcher(launcher)

        launcher.set_can_raise_exception(self.can_raise_exception)
        launcher.set_concurrent(self.concurrent)

        launch_domain = strategy.launch_domain if strategy.parallel else None
        self._add_communicators(launcher, launch_domain)

        result: Union[Future, FutureMap]
        if launch_domain is not None:
            result = launcher.execute(launch_domain)
        else:
            result = launcher.execute_single()

        self._demux_scalar_stores(result, launch_domain)


class ManualTask(Operation, Task):
    """
    A type of tasks that need explicit parallelization
    """

    def __init__(
        self,
        context: Context,
        task_id: int,
        launch_domain: Rect,
        mapper_id: int,
        op_id: int,
    ) -> None:
        super().__init__(
            context=context,
            task_id=task_id,
            mapper_id=mapper_id,
            op_id=op_id,
        )
        self._launch_domain: Rect = launch_domain
        self._input_projs: list[Union[ProjFn, None]] = []
        self._output_projs: list[Union[ProjFn, None]] = []
        self._reduction_projs: list[Union[ProjFn, None]] = []

        self._input_parts: list[StorePartition] = []
        self._output_parts: list[Union[StorePartition, None]] = []
        self._reduction_parts: list[tuple[StorePartition, int]] = []

    @property
    def launch_ndim(self) -> int:
        return self._launch_domain.dim

    def get_all_stores(self) -> OrderedSet[Store]:
        return OrderedSet()

    @staticmethod
    def _check_arg(arg: Union[Store, StorePartition]) -> None:
        if not isinstance(arg, (Store, StorePartition)):
            raise ValueError(
                f"Expected a Store or StorePartition, but got {type(arg)}"
            )

    def add_input(
        self,
        arg: Union[Store, StorePartition],
        proj: Optional[ProjFn] = None,
    ) -> None:
        """
        Adds a store as input to the task

        Parameters
        ----------
        arg : Store or StorePartition
            Store or store partition to pass as input
        proj : ProjFn, optional
            Projection function
        """
        self._check_arg(arg)
        if isinstance(arg, Store):
            self._input_parts.append(arg.partition(REPLICATE))
        else:
            self._input_parts.append(arg)
        self._input_projs.append(proj)

    def add_output(
        self,
        arg: Union[Store, StorePartition],
        proj: Optional[ProjFn] = None,
    ) -> None:
        """
        Adds a store as output to the task

        Parameters
        ----------
        arg : Store or StorePartition
            Store or store partition to pass as output
        proj : ProjFn, optional
            Projection function

        Raises
        ------
        NotImplementedError
            If the store is unbound
        """
        self._check_arg(arg)
        if isinstance(arg, Store):
            if arg.kind is Future:
                self._scalar_outputs.append(len(self._outputs))
            elif arg.unbound:
                if arg.ndim != self.launch_ndim:
                    raise NotImplementedError(
                        "Unbound store with an incompatible number of "
                        "dimensions cannot be used with manually parallelized "
                        "task"
                    )
                self._unbound_outputs.append(len(self._outputs))
            self._outputs.append(arg)
            if arg.unbound:
                # FIXME: Need better placeholders for unbound stores
                self._output_parts.append(None)
            else:
                self._output_parts.append(arg.partition(REPLICATE))
        else:
            self._output_parts.append(arg)
        self._output_projs.append(proj)

    def add_reduction(
        self,
        arg: Union[Store, StorePartition],
        redop: int,
        proj: Optional[ProjFn] = None,
    ) -> None:
        """
        Adds a store to the task for reduction

        Parameters
        ----------
        arg : Store or StorePartition
            Store or store partition to pass for reduction
        proj : ProjFn, optional
            Projection function
        """
        self._check_arg(arg)
        if isinstance(arg, Store):
            if arg.kind is Future:
                self._scalar_reductions.append(len(self._reductions))
                self._reductions.append((arg, redop))
            self._reduction_parts.append((arg.partition(REPLICATE), redop))
        else:
            self._reduction_parts.append((arg, redop))
        self._reduction_projs.append(proj)

    def add_alignment(self, store1: Store, store2: Store) -> None:
        raise TypeError(
            "Partitioning constraints are not allowed for "
            "manually parallelized tasks"
        )

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        raise TypeError(
            "Partitioning constraints are not allowed for "
            "manually parallelized tasks"
        )

    def add_constraint(self, constraint: Constraint) -> None:
        raise TypeError(
            "Partitioning constraints are not allowed for "
            "manually parallelized tasks"
        )

    def launch(self, strategy: Strategy) -> None:
        tag = self.context.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG
        launcher = TaskLauncher(
            self.context,
            self._task_id,
            self.mapper_id,
            tag=tag,
            error_on_interference=False,
            side_effect=self._side_effect,
            provenance=self.provenance,
        )

        for part, proj_fn in zip(self._input_parts, self._input_projs):
            req = part.get_requirement(self.launch_ndim, proj_fn)
            launcher.add_input(part.store, req, tag=0)

        for opart, proj_fn in zip(self._output_parts, self._output_projs):
            if opart is None:
                continue
            req = opart.get_requirement(self.launch_ndim, proj_fn)
            launcher.add_output(opart.store, req, tag=0)

        for (part, redop), proj_fn in zip(
            self._reduction_parts, self._reduction_projs
        ):
            req = part.get_requirement(self.launch_ndim, proj_fn)
            req.redop = part.store.type.reduction_op_id(redop)
            can_read_write = part.is_disjoint_for(self._launch_domain)
            launcher.add_reduction(
                part.store, req, tag=0, read_write=can_read_write
            )

        for store, proj_fn in zip(self._outputs, self._output_projs):
            if not store.unbound:
                continue
            # TODO: Need an interface for clients to specify isomorphism
            # bewteen unbound stores
            fspace = self._context.runtime.create_field_space()
            field_id = fspace.allocate_field(store.type)
            launcher.add_unbound_output(store, fspace, field_id)

        self._add_scalar_args_to_launcher(launcher)

        launcher.set_can_raise_exception(self.can_raise_exception)
        launcher.set_concurrent(self.concurrent)

        self._add_communicators(launcher, self._launch_domain)

        result = launcher.execute(self._launch_domain)

        self._demux_scalar_stores(result, self._launch_domain)


class Copy(AutoOperation):
    """
    A special kind of operation for copying data from one store to another.
    """

    def __init__(
        self,
        context: Context,
        mapper_id: int,
        op_id: int,
    ) -> None:
        super().__init__(context=context, mapper_id=mapper_id, op_id=op_id)
        self._source_indirects: list[Store] = []
        self._target_indirects: list[Store] = []
        self._source_indirect_parts: list[PartSym] = []
        self._target_indirect_parts: list[PartSym] = []
        self._source_indirect_out_of_range = True
        self._target_indirect_out_of_range = True

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Copy(uid:{self._op_id})"

    @property
    def inputs(self) -> list[Store]:
        return super().inputs + self._source_indirects + self._target_indirects

    def add_input(self, store: Store) -> None:
        """
        Adds a store as a source of the copy

        Parameters
        ----------
        store : Store
            Source store

        Raises
        ------
        ValueError
            If the store is scalar or unbound
        """
        if store.kind is Future or store.unbound:
            raise ValueError(
                "Copy input must be a normal, region-backed store"
            )
        self._check_store(store)
        partition = self._get_unique_partition(store)
        self._inputs.append(store)
        self._input_parts.append(partition)

    def add_output(self, store: Store) -> None:
        """
        Adds a store as a target of the copy. To avoid ambiguity in matching
        sources and targets, one copy cannot have both normal targets and
        reduction targets.

        Parameters
        ----------
        store : Store
            Target store

        Raises
        ------
        RuntimeError
            If the copy already has a reduction target
        ValueError
            If the store is scalar or unbound
        """
        if len(self._reductions) > 0:
            raise RuntimeError(
                "Copy targets must be either all normal outputs or reductions"
            )
        if store.kind is Future or store.unbound:
            raise ValueError(
                "Copy target must be a normal, region-backed store"
            )

        self._check_store(store)
        partition = self._get_unique_partition(store)
        self._outputs.append(store)
        self._output_parts.append(partition)

    def add_reduction(self, store: Store, redop: int) -> None:
        """
        Adds a store as a reduction target of the copy. To avoid ambiguity in
        matching sources and targets, one copy cannot have both normal targets
        and reduction targets.

        Parameters
        ----------
        store : Store
            Reduction target store
        redop : int
            Reduction operator ID

        Raises
        ------
        RuntimeError
            If the copy already has a normal target
        ValueError
            If the store is scalar or unbound
        """
        if len(self._outputs) > 0:
            raise RuntimeError(
                "Copy targets must be either all normal outputs or reductions"
            )
        if store.kind is Future or store.unbound:
            raise ValueError(
                "Copy target must be a normal, region-backed store"
            )
        self._check_store(store)
        partition = self._get_unique_partition(store)
        self._reductions.append((store, redop))
        self._reduction_parts.append(partition)

    def add_source_indirect(self, store: Store) -> None:
        """
        Adds an indirection for sources. A copy can have only up to one source
        indirection.

        Parameters
        ----------
        store : Store
            Source indirection store

        Raises
        ------
        RuntimeError
            If the copy already has a source indirection
        """
        if len(self._source_indirects) != 0:
            raise RuntimeError(
                "There can be only up to one source indirection store for "
                "a Copy operation"
            )
        self._check_store(store)
        partition = self._get_unique_partition(store)
        self._source_indirects.append(store)
        self._source_indirect_parts.append(partition)

    def add_target_indirect(self, store: Store) -> None:
        """
        Adds an indirection for targets. A copy can have only up to one target
        indirection.

        Parameters
        ----------
        store : Store
            Target indirection store

        Raises
        ------
        RuntimeError
            If the copy already has a target indirection
        """
        if len(self._target_indirects) != 0:
            raise RuntimeError(
                "There can be only up to one target indirection store for "
                "a Copy operation"
            )
        self._check_store(store)
        partition = self._get_unique_partition(store)
        self._target_indirects.append(store)
        self._target_indirect_parts.append(partition)

    def set_source_indirect_out_of_range(self, flag: bool) -> None:
        self._source_indirect_out_of_range = flag

    def set_target_indirect_out_of_range(self, flag: bool) -> None:
        self._target_indirect_out_of_range = flag

    @property
    def constraints(self) -> list[Constraint]:
        constraints: list[Constraint] = []
        if len(self._source_indirects) + len(self._target_indirects) == 0:
            for src, tgt in zip(self._input_parts, self._output_parts):
                if src.store.shape != tgt.store.shape:
                    raise ValueError(
                        "Each output must have the same shape as the "
                        f"input, but got {tuple(src.store.shape)} and "
                        f"{tuple(tgt.store.shape)}"
                    )
                constraints.append(src == tgt)
        else:
            if len(self._source_indirects) > 0:
                output_parts = (
                    self._output_parts
                    if len(self._outputs) > 0
                    else self._reduction_parts
                )
                for src, tgt in zip(self._source_indirect_parts, output_parts):
                    if src.store.shape != tgt.store.shape:
                        raise ValueError(
                            "Each output must have the same shape as the "
                            "corresponding source indirect field, but got "
                            f"{tuple(src.store.shape)} and "
                            f"{tuple(tgt.store.shape)}"
                        )
                    constraints.append(src == tgt)
            if len(self._target_indirects) > 0:
                for src, tgt in zip(
                    self._input_parts, self._target_indirect_parts
                ):
                    if src.store.shape != tgt.store.shape:
                        raise ValueError(
                            "Each input must have the same shape as the "
                            "corresponding target indirect field, but got "
                            f"{tuple(src.store.shape)} and "
                            f"{tuple(tgt.store.shape)}"
                        )
                    constraints.append(src == tgt)
        return constraints

    def add_alignment(self, store1: Store, store2: Store) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for copies"
        )

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for copies"
        )

    def add_constraint(self, constraint: Constraint) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for copies"
        )

    def launch(self, strategy: Strategy) -> None:
        launcher = CopyLauncher(
            self.context,
            source_oor=self._source_indirect_out_of_range,
            target_oor=self._target_indirect_out_of_range,
            mapper_id=self.mapper_id,
            provenance=self.provenance,
        )

        assert len(self._inputs) == len(self._outputs) or len(
            self._inputs
        ) == len(self._reductions)
        assert len(self._source_indirects) == 0 or len(
            self._source_indirects
        ) == len(self._inputs)
        assert len(self._target_indirects) == 0 or len(
            self._target_indirects
        ) == len(self._outputs)

        # FIXME: today a copy is a scatter copy only when a target indirection
        # is given. In the future, we may pass store transforms directly to
        # Legion and some transforms turn some copies into scatter copies even
        # when no target indirection is provided. So, this scatter copy check
        # will need to be extended accordingly.
        scatter = len(self._target_indirects) > 0

        for store, part_symb in zip(self._inputs, self._input_parts):
            req, tag, _ = self.get_requirement(store, part_symb, strategy)
            launcher.add_input(store, req, tag=tag)

        for store, part_symb in zip(self._outputs, self._output_parts):
            assert not store.unbound
            req, tag, store_part = self.get_requirement(
                store, part_symb, strategy
            )
            if scatter:
                launcher.add_inout(store, req, tag=tag)
            else:
                launcher.add_output(store, req, tag=tag)

        for (store, redop), part_symb in zip(
            self._reductions, self._reduction_parts
        ):
            req, tag, store_part = self.get_requirement(
                store, part_symb, strategy
            )
            req.redop = store.type.reduction_op_id(redop)
            launcher.add_reduction(store, req, tag=tag)
        for store, part_symb in zip(
            self._source_indirects, self._source_indirect_parts
        ):
            req, tag, store_part = self.get_requirement(
                store, part_symb, strategy
            )
            launcher.add_source_indirect(store, req, tag=tag)
        for store, part_symb in zip(
            self._target_indirects, self._target_indirect_parts
        ):
            req, tag, store_part = self.get_requirement(
                store, part_symb, strategy
            )
            launcher.add_target_indirect(store, req, tag=tag)

        if strategy.launch_domain is not None:
            launcher.execute(strategy.launch_domain)
        else:
            launcher.execute_single()


class Fill(AutoOperation):
    """
    A special kind of operation for filling a store with constant values
    """

    def __init__(
        self,
        context: Context,
        lhs: Store,
        value: Store,
        mapper_id: int,
        op_id: int,
    ) -> None:
        super().__init__(context=context, mapper_id=mapper_id, op_id=op_id)
        if not value.scalar:
            raise ValueError("Fill value must be a scalar Store")
        if lhs.unbound:
            raise ValueError("Fill lhs must be a bound Store")
        if lhs.kind is Future:
            raise ValueError("Fill lhs must be a RegionField-backed Store")
        self._add_value(value)
        self._add_lhs(lhs)

    def _add_value(self, value: Store) -> None:
        partition = self._get_unique_partition(value)
        self._inputs.append(value)
        self._input_parts.append(partition)

    def _add_lhs(self, lhs: Store) -> None:
        partition = self._get_unique_partition(lhs)
        self._outputs.append(lhs)
        self._output_parts.append(partition)

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Fill(uid:{self._op_id})"

    def add_alignment(self, store1: Store, store2: Store) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for fills"
        )

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for fills"
        )

    def add_constraint(self, constraint: Constraint) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for fills"
        )

    def launch(self, strategy: Strategy) -> None:
        lhs = self._outputs[0]
        lhs_part_sym = self._output_parts[0]
        lhs_proj, _, lhs_part = self.get_requirement(
            lhs, lhs_part_sym, strategy
        )
        lhs.set_key_partition(lhs_part.partition)
        launcher = FillLauncher(
            self.context,
            lhs,
            lhs_proj,
            self._inputs[0],
            mapper_id=self.mapper_id,
            provenance=self.provenance,
        )
        if strategy.launch_domain is not None:
            launcher.execute(strategy.launch_domain)
        else:
            launcher.execute_single()


class _RadixProj:
    def __init__(self, radix: int, offset: int) -> None:
        self._radix = radix
        self._offset = offset

    def __call__(self, p: SymbolicPoint) -> ProjOut:
        return (p[0] * self._radix + self._offset,)


class Reduce(AutoOperation):
    def __init__(
        self,
        context: Context,
        task_id: int,
        radix: int,
        mapper_id: int,
        op_id: int,
    ) -> None:
        super().__init__(
            context=context,
            mapper_id=mapper_id,
            op_id=op_id,
        )
        self._runtime = context.runtime
        self._radix = radix
        self._task_id = task_id

    def add_input(self, store: Store) -> None:
        self._check_store(store)
        partition = self._get_unique_partition(store)
        self._inputs.append(store)
        self._input_parts.append(partition)

    def add_output(self, store: Store) -> None:
        assert store.unbound
        partition = self._get_unique_partition(store)
        self._unbound_outputs.append(len(self._outputs))
        self._outputs.append(store)
        self._output_parts.append(partition)

    def launch(self, strategy: Strategy) -> None:
        assert len(self._inputs) == 1 and len(self._outputs) == 1

        result = self._outputs[0]

        output = self._inputs[0]
        opart = output.partition(strategy.get_partition(self._input_parts[0]))

        done = False
        launch_domain = None
        fan_in = 1
        if strategy.parallel:
            assert strategy.launch_domain is not None
            launch_domain = strategy.launch_domain
            fan_in = launch_domain.get_volume()

        proj_fns = list(
            _RadixProj(self._radix, off) for off in range(self._radix)
        )

        while not done:
            input = output
            ipart = opart

            tag = self.context.core_library.LEGATE_CORE_TREE_REDUCE_TAG
            launcher = TaskLauncher(
                self.context,
                self._task_id,
                self.mapper_id,
                tag=tag,
                provenance=self.provenance,
            )

            for proj_fn in proj_fns:
                launcher.add_input(input, ipart.get_requirement(1, proj_fn))

            output = self._context.create_store(input.type)
            fspace = self._runtime.create_field_space()
            field_id = fspace.allocate_field(input.type)
            launcher.add_unbound_output(output, fspace, field_id)

            num_tasks = (fan_in + self._radix - 1) // self._radix
            launch_domain = Rect([num_tasks])
            weights = launcher.execute(launch_domain)

            launch_shape = Shape(c + 1 for c in launch_domain.hi)
            weighted = Weighted(launch_shape, weights)
            output.set_key_partition(weighted)
            opart = output.partition(weighted)

            fan_in = num_tasks
            done = fan_in == 1

        result.set_storage(output.storage)
