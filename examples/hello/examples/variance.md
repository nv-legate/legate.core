<!--
Copyright 2023 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

# Variance Example

The code for this example can be found in the [library file](../hello/hello.py) and [example](variance.py).

## Creating a store

As seen in the `iota` task, a store can be created from a context as, e.g.

```
output = user_context.create_store(
    types.float32,
    shape=(size,),
    optimize_scalar=True,
)
```

At this point, the store may not be allocated or contain data,
but can still be passed to tasks as a valid output handle.

## Elementwise task with aligned partitions

Tasks are also created on a context:

```
task = user_context.create_auto_task(HelloOpCode.SQUARE)

task.add_input(input)
task.add_output(output)
task.add_alignment(input, output)
task.execute()
```

An auto task indicates Legate should auto-partition based
on cost heuristics and partitioning constraints.
An input and output array are added.
The most critical step here, though, is the alignment of
the input and output. Since we want to do elementwise operations,
we need the input and output partitions to be aligned.
This expresses an auto-partitioning constraint.
Finally, the task is enqueued by calling its `execute` method.

## Reduction (Summation)

We similarly set up a task, but now add the output
as a reduction.

```
task = user_context.create_auto_task(HelloOpCode.SUM)

task.add_input(input)
task.add_reduction(output, types.ReductionOp.ADD)
task.execute()
```

The output is a scalar, which means there is no partitioning
alignment constraint with input and output.

## Using data from other Legate libraries

Data structures from other libraries (e.g. cunumeric)
can be passed into functions from other Legate libraries,
even if the libraries are unaware of each other.
Legate provides a common interface for data structures
to provide a schema and access to its underlying stores.
This is shown in the `_get_legate_store` function via
the `__legate_data_interface__`.

