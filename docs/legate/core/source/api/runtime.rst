.. _label_runtime:

.. currentmodule:: legate.core

Runtime and Library Contexts
============================

Library
-------

A ``Library`` class is an interface that every library descriptor needs to
implement. Each library should tell the Legate runtime how to initialize and
configure the library, and this class provides a common way to reveal that
information to the runtime. Each library should register to the runtime a
library descriptor object that implements ``Library`` directly or via duck
typing. (See :meth:`legate.core.runtime.Runtime.register_library`.)

.. autosummary::
   :toctree: generated/

   Library.get_name
   Library.get_shared_library
   Library.get_c_header
   Library.get_registration_callback
   Library.get_resource_configuration


Resource configuration
----------------------

A ``ResourceConfig`` object describes the maximum number of handles that a
library uses.

.. autosummary::
   :toctree: generated/

   ResourceConfig.max_tasks
   ResourceConfig.max_reduction_ops
   ResourceConfig.max_mappers


Context
-------

A ``Context`` object provides APIs for creating stores and issuing tasks and
other kinds of operations. When a library registers itself to the Legate
runtime, the runtime gives back a context object unique to the library.

.. autosummary::
   :toctree: generated/

   context.Context.create_store
   context.Context.create_manual_task
   context.Context.create_auto_task
   context.Context.create_copy
   context.Context.create_fill
   context.Context.issue_execution_fence
   context.Context.tree_reduce
   context.Context.get_tunable
   context.Context.provenance
   context.Context.annotation
   context.Context.set_provenance
   context.Context.reset_provenance
   context.Context.push_provenance
   context.Context.pop_provenance
   context.Context.track_provenance


Legate Runtime
--------------

.. autosummary::
   :toctree: generated/

   runtime.Runtime.num_cpus
   runtime.Runtime.num_omps
   runtime.Runtime.num_gpus
   runtime.Runtime.register_library
   runtime.Runtime.create_future


Annotation
----------

An ``Annotation`` is a context manager to set library specific annotations that
are to be attached to operations issued within a scope. A typical usage of
``Annotation`` would look like this:

::

  with Annotation(lib_context, { "key1" : "value1", "key2" : "value2", ... }:
    ...

Then each operation in the scope is annotated with the key-value pairs,
which are later rendered in execution profiles.

.. autosummary::
   :toctree: generated/

   context.Annotation.__init__
