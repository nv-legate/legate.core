.. currentmodule:: legate.core.store

Store
=====

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
   Store.extents


Transformation
--------------

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

   Store.storage
   Store.has_storage
   Store.get_inline_allocation


Partition management
--------------------

.. autosummary::
   :toctree: generated/

   Store.get_key_partition
   Store.has_key_partition
   Store.set_key_partition
   Store.reset_key_partition
   Store.compute_key_partition
   Store.find_restrictions
   Store.partition_by_tiling
