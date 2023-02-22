.. _label_operation:

.. currentmodule:: legate.core.operation

Tasks and Other Operation Kinds
===============================

AutoTask
--------

.. autosummary::
   :toctree: generated/

   AutoTask.side_effect
   AutoTask.set_concurrent
   AutoTask.set_side_effect
   AutoTask.add_input
   AutoTask.add_output
   AutoTask.add_reduction
   AutoTask.add_scalar_arg
   AutoTask.throws_exception
   AutoTask.can_raise_exception
   AutoTask.declare_partition
   AutoTask.add_constraint
   AutoTask.add_alignment
   AutoTask.add_broadcast
   AutoTask.add_nccl_communicator
   AutoTask.add_cpu_communicator
   AutoTask.execute


ManualTask
----------

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

Copy
----

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

.. autosummary::
   :toctree: generated/

   Fill.execute
