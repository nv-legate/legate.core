.. currentmodule:: legate.core.store

Store
=====

`Store` is a multi-dimensional data container for fixed-size elements. Stores
are internally partitioned and distributed across the system. By default,
Legate clients need not create nor maintain the partitions explicitly, and the
Legate runtime is responsible for managing them. Legate clients can control how
stores should be partitioned for a given task by attaching partitioning
constraints to the task (see section :ref:`label_operation` for partitioning
constraint APIs).

Each Store object is a logical handle to the data and is not immediately
associated with a physical allocation. To access the data, a client must
`map` the store to a physical instance. A client can map a store by passing
it to a task, in which case the task body can see the allocation, or calling
``get_inline_allocation``, which gives the client a linear handle to the
physical allocation (see section :ref:`label_allocation` for details about
inline allocations).

Normally, a store gets a fixed shape upon creation. However, there is a special
type of stores called `unbound` stores whose shapes are unknown at creation
time. (see section :ref:`label_runtime` for the store creation API.) The shape
of an unbound store is determined by a task that first updates the store; upon
the submission of the task, the store becomes a normal store. Passing an
unbound store as a read-only argument or requesting an inline allocation of an
unbound store are invalid.

One consequence due to the nature of unbound stores is that querying the shape
of a previously unbound store can block the client's control flow for an
obvious reason; to know the shape of the store whose shape was unknown at
creation time, the client must wait until the updater task to finish. However,
passing a previously unbound store to a downstream operation can be
non-blocking, as long as the operation requires no changes in the partitioning
and mapping for the store.


Basic Properties
----------------

.. autosummary::
   :toctree: generated/

   Store.shape
   Store.ndim
   Store.size
   Store.type
   Store.kind
   Store.unbound
   Store.scalar
.. Store.extents


Transformation
--------------

Legate provides several API calls to transform stores. A store after a
transformation is a view to the original store; i.e., any changes made to the
transformed store are visible via the original one and vice versa.

.. autosummary::
   :toctree: generated/

   Store.transform
   Store.transformed
   Store.promote
   Store.project
   Store.slice
   Store.transpose
   Store.delinearize


Storage management
------------------

.. autosummary::
   :toctree: generated/

   Store.get_inline_allocation
.. Store.storage
.. Store.has_storage


Partition management
--------------------

In most cases, Legate clients need not create nor manage partitions manually by
themselves. However, there are occasions where the clients need to parallelize
tasks manually, for which stores need to be partitioned manually as well. For
those occasions, clients may want to query and update the `key` partition of
each store, i.e., the partition used for updating the store for the last time.
The following are the API calls for manual partition management.

.. autosummary::
   :toctree: generated/

   Store.get_key_partition
   Store.set_key_partition
   Store.reset_key_partition
   Store.partition_by_tiling


.. _label_store_partition:

StorePartition
==============

A ``StorePartition`` is an object that represents a partitioned state of a
store. A store partition is a name of a collection of `sub-stores`, each of
which contains to a subset of elements in the store. Sub-stores in a store
partition are uniquely identified by their `colors`, and a set of all colors
of a given store partition is called a `color space`.

It is recommended that store partitions and their sub-stores be used as
arguments to ``ManualTask`` (see section :ref:`label_operation` for APIs for
manual parallelization).


.. autosummary::
   :toctree: generated/

   StorePartition.store
   StorePartition.partition
   StorePartition.get_child_store
