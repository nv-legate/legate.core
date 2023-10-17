.. _label_machine:

.. currentmodule:: legate.core

Machine and Resource Scoping
============================

By default, each Legate operation is allowed to use the entire machine for
parallelization, but oftentimes client programs want control on the machine
resource assigned to each section of the program. Legate provides a
programmatic way to control resource assignment, called a *resource scoping*.

To scope the resource, a client takes two steps. First, the client queries the
machine resource available in the given scope and shrinks it to a subset to
assign to a scope. Then, the client assigns that subset of the machine to a scope
with a  usual ``with`` statement. All Legate operations issued within that
``with`` block are now subject to the resource scoping. The steps look like the
following pseudocode:

::

  # Retrieves the machine of the current scope
  machine = legate.core.get_machine()
  # Extracts a subset to assign to a nested scope
  subset = extract_subset(machine)
  # Installs the new machine to a scope
  with subset:
    ...

The machine available to a nested scope is always a subset of that for the
outer scope. If the machine given to a scope has some resources that are not
part of the machine for the outer scope, they will be removed during the
resource scoping. The machine used in a scoping must not be empty; otherwise,
an ``EmptyMachineError`` will be raised.

In cases where a machine has more than one kind of processor, the
parallelization heuristic has the following precedence on preference between
different types: GPU > OpenMP > CPU.

Metadata about the machine is stored in a ``Machine`` object and the
``Machine`` class provides APIs for querying and subdivision of resources.

Machine
-------

.. autosummary::
   :toctree: generated/

   Machine.preferred_kind
   Machine.kinds
   Machine.get_processor_range
   Machine.get_node_range
   Machine.only
   Machine.count
   Machine.empty
   Machine.__and__
   Machine.__len__
   Machine.__getitem__


ProcessorRange
--------------

A ``ProcessorRange`` is a half-open interval of global processor IDs.

.. autosummary::
   :toctree: generated/

   ProcessorRange.empty
   ProcessorRange.get_node_range
   ProcessorRange.slice
   ProcessorRange.__and__
   ProcessorRange.__len__
   ProcessorRange.__getitem__
