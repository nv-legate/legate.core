.. _label_runtime:

.. currentmodule:: legate.core

Runtime and Library Contexts
============================

Legate Runtime
--------------

.. autosummary::
   :toctree: generated/

   runtime.Runtime.core_context
   runtime.Runtime.core_library
   runtime.Runtime.num_cpus
   runtime.Runtime.num_omps
   runtime.Runtime.num_gpus
   runtime.Runtime.register_library
   runtime.Runtime.create_future


Library
-------

.. autosummary::
   :toctree: generated/

   Library.get_name
   Library.get_shared_library
   Library.get_c_header
   Library.get_registration_callback
   Library.get_resource_configuration


Resource configuration
----------------------

.. autosummary::
   :toctree: generated/

   ResourceConfig.max_tasks


Context
-------

.. autosummary::
   :toctree: generated/

   context.Context.runtime
   context.Context.library
   context.Context.annotation
   context.Context.provenance
   context.Context.get_tunable
   context.Context.set_provenance
   context.Context.reset_provenance
   context.Context.push_provenance
   context.Context.pop_provenance
   context.Context.track_provenance
   context.Context.create_task
   context.Context.create_manual_task
   context.Context.create_auto_task
   context.Context.create_copy
   context.Context.create_fill
   context.Context.create_store
   context.Context.issue_execution_fence
   context.Context.tree_reduce

Annotation
----------

.. autosummary::
   :toctree: generated/

   context.Annotation.__init__
