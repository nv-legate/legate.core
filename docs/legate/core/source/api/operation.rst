.. _label_operation:

.. currentmodule:: legate.core.operation

Operations
==========

Operations in Legate are by default automatically parallelized. Legate extracts
parallelism from an operation by partitioning its store arguments. Operations
usually require the partitions to be aligned in some way; e.g., partitioning
vectors across multiple addition tasks requires the vectors to be partitioned
in the same way. Legate provides APIs for developers to control how stores are
partitioned via `partitioning constraints`.

When an operation needs a store to be partitioned more than one way, the
operation can create `partition symbols` and use them in partitioning
constraints. In that case, a partition symbol must be passed along with the
store when the store is added. Stores can be partitioned in multiple ways when
they are used only for read accesses or reductions.

AutoTask
--------

``AutoTask`` is a type of tasks that are automatically parallelized. Each
Legate task is associated with a task id that uniquely names a task to invoke.
The actual task implementation resides on the C++ side.

.. autosummary::
   :toctree: generated/

   AutoTask.add_input
   AutoTask.add_output
   AutoTask.add_reduction
   AutoTask.add_scalar_arg
   AutoTask.declare_partition
   AutoTask.add_constraint
   AutoTask.add_alignment
   AutoTask.add_broadcast
   AutoTask.throws_exception
   AutoTask.can_raise_exception
   AutoTask.add_nccl_communicator
   AutoTask.add_cpu_communicator
   AutoTask.side_effect
   AutoTask.set_concurrent
   AutoTask.set_side_effect
   AutoTask.execute


Copy
----

``Copy`` is a special kind of operation for copying data from one store to
another. Unlike tasks that are mapped to and run on application processors,
copies are performed by the DMA engine in the runtime. Also, unlike tasks that
are user-defined, copies have well-defined semantics and come with predefined
partitioning assumptions on stores. Hence, copies need not take partitioning
constraints from developers.

A copy can optionally take a store for indices that need to be used in
accessing the source or target. With an `indirection` store on the source, the
copy performs a gather operation, and with an indirection on the target, the
copy does a scatter; when indirections exist for both the source and target,
the copy turns into a full gather-scatter copy. Out-of-bounds indices are not
checked and can produce undefined behavior. The caller therefore is responsible
for making sure the indices are within bounds.

.. autosummary::
   :toctree: generated/

   Copy.add_input
   Copy.add_output
   Copy.add_reduction
   Copy.add_source_indirect
   Copy.add_target_indirect
   Copy.execute

Fill
----

``Fill`` is a special kind of operation for filling a store with constant values.
Like coipes, fills are performed by the DMA engine and their partitioning
constraints are predefined.

.. autosummary::
   :toctree: generated/

   Fill.execute


Manually Parallelized Tasks
---------------------------

In some occassions, tasks are unnatural or even impossible to write in the
auto-parallelized style. For those occassions, Legate provides explicit control
on how tasks are parallelized via ``ManualTask``. Each manual task requires the
caller to provide a `launch domain` that determines the degree of parallelism
and also names task instances initiaed by the task. Direct store arguments to a
manual task are assumed to be replicated across task instances, and it's the
developer's responsibility to partition stores. Mapping between points in the
launch domain and colors in the color space of a store partition is assumed to
be an identity mapping by default, but it can be configured with a `projection
function`, a Python function on tuples of coordinates. (See
:ref:`StorePartition <label_store_partition>` for definitions of color,
color space, and store partition.)

.. autosummary::
   :toctree: generated/

   ManualTask.side_effect
   ManualTask.set_concurrent
   ManualTask.set_side_effect
   ManualTask.add_input
   ManualTask.add_output
   ManualTask.add_reduction
   ManualTask.add_scalar_arg
   ManualTask.throws_exception
   ManualTask.can_raise_exception
   ManualTask.add_nccl_communicator
   ManualTask.add_cpu_communicator
   ManualTask.execute
