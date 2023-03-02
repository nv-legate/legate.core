.. _label_allocation:

.. currentmodule:: legate.core.allocation

Inline mapping
==============

When a client requests an immediate allocation of a store with
:meth:`legate.core.store.Store.get_inline_allocation`, the runtime gives you
back an ``InlineMappedAllocation`` object, which is a thin wrapper around the
allocation. Since the runtime needs to keep track of lifetimes of Python
objects using the allocation, the wrapper reveals the allocation to a callback
and not directly. Doing it this way allows the runtime to capture the object
constructed from the allocation and tie their lifetimes.


.. autosummary::
   :toctree: generated/

   InlineMappedAllocation.consume
