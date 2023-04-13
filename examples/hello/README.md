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
# Legate Hello World Application

Here we illustrate a minimal example to get a Legate library up and running.
The example here shows how to get started with the minimum amount of boilerplate.
For advanced use cases, the boilerplate generated can be customized as needed.
In general, a Legate application will need to implement three pieces.

1. Build system
1. C++ tasks
1. Python library

Please refer to the README in the [Legate repo](https://github.com/nv-legate/legate.core/blob/HEAD/README.md)
for first installing `legate.core`.  We strongly recommend creating a Conda environment for development and testing.

# Build System

## Build Steps

To build the project, the user can do the following:

```
$ cmake -S . -B build
$ cmake --build build
$ python -m pip install -e .
```

This performs an editable install of the project, which we recommend for development.
If `cmake` fails to find Legate, the path to the installed Legate can be manually
specific as `-Dlegate_core_ROOT=<...>` to the `cmake` configuration.
Alternatively, the user can just do a regular pip installation:

```
$ python -m pip install .
```

These approaches are illustrated in the `editable-install.sh` and `install.sh` scripts.
In particular, `editable-install.sh` shows how to use Legate install info to
point CMake to the correct installation root.

## CMake
CMake is the officially supported mechanism for building Legate libraries.
Legate exports a CMake target and helper functions for building libraries and provides by-far the easiest onboarding.
There are only a few main steps in setting up a build system.
First, the user should initialize a CMake project.

```cmake
cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(hello VERSION 1.0 LANGUAGES C CXX)
```

Next the user needs to find an existing Legate core:

```cmake
find_package(legate_core REQUIRED)
```

Once the `legate_core` package is located, a number of helper functions will be available.
In a source folder, the user can define a library that will implement the C++ tasks:

```cmake
legate_cpp_library_template(hello TEMPLATE_SOURCES)

add_library(
  hello
  hello_world.cc
  hello_world.h
  ${TEMPLATE_SOURCES}
)
target_link_libraries(hello PRIVATE legate::core)
```

First, a helper function is invoked to generate the Legate C++ boilerplate files.
The list of generated files is returned in the `TEMPLATE_SOURCES` variable.
Second, the CMake library is linked against the imported `legate::core` target.

Two helper functions are provided to generate the Python boilerplate.
In the top-level CMakeLists.txt, the Python-C bindings can be generated using CFFI:

```cmake
legate_add_cffi(${CMAKE_SOURCE_DIR}/src/hello_world.h TARGET hello)
```

The header file is implemented by the user and contains all the enums required
to implement a Legate library. The necessary Python file is generated in the `hello`
subdirectory. Additionally, the user may want to generate a standard `library.py`
in the Python `hello` folder:

```cmake
legate_python_library_template(hello)
```

Finally, default pip installation hooks (via scikit-build) can be added:

```cmake
legate_default_python_install(hello EXPORT hello-export)
```

## Editable Builds

Although the final user Python library will likely be installed with `pip`,
the user will usually need to iterate on the C++ implementation of tasks
for debugging and optmization.  The user will therefore want to be able
to first build the C++ pieces of the project and then install the Python.
To support this workflow, legate provides a helper function:

```cmake
legate_add_cpp_subdirectory(src hello EXPORT hello-export)
```
This encapsulates the build target `hello` so that the C++ library can
be first built with CMake and then pip-installed in a separate step.
This is optional, though, and the entire build can always be executed by
doing a regular pip install:

```
$ python -m pip install .
```

# C++ tasks

First, a `hello_world.h` header is needed to define all enums. In this case,
we have enums identifying the different task types:

```cpp
enum HelloOpCode {
  _OP_CODE_BASE = 0,
  HELLO_WORLD_TASK = 1,
};
```

We implement this CPU-only task in a `hello_world.cc`.

```cpp
#include "legate_library.h"
#include "hello_world.h"

namespace hello {
```

The source file should include the library header and the generated file `legate_library.h`.
Because the target was named `hello` in the build files, all generated files create types
in the `hello` namespace.

The task implementation is simple:

```cpp
class HelloWorldTask : public Task<HelloWorldTask, HELLO_WORLD_TASK> {
 public:
  static void cpu_variant(legate::TaskContext& context){
    std::string message = context.scalars()[0].value<std::string>();
    std::cout << message << std::endl;
  }
};
```
Here we define a CPU variant. The task is given the unique enum ID from `hello_world.h`.
The task unpacks a string from the input context and prints it.
Task types needed to be statically registered, which requires a bit of extra boilerplate:

```cpp
namespace
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::HelloWorldTask::register_variants();
}

}
```

Any tasks instantiated in the Python library will ultimately invoke this C++ task.

# Python library

The example uses two generated files `library.py` and `install_info.py`.
The implementation of tasks is provided in the `hello.py` file.
First, we have to import a few types and a context object for creating tasks.
The context object is automatically created in the generated boilerplate.

```python
from .library import user_context, user_lib
from enum import IntEnum
from legate.core import Rect
import legate.core.types as types
```

The C++ enums can be mapped into Python:

```python
class HelloOpCode(IntEnum):
    HELLO_WORLD = user_lib.cffi.HELLO_WORLD_TASK
```

The example here provides two library functions. The first prints a single message.
The second prints a fixed number of of messages. For `print_hello`,
a new task is created in `user_context`. The message string is added as a scalar argument.
In the second example, a launch domain for a fixed `n` is provided.

These library functions can now be imported and used in python.
This is shown in `examples/hello.py`:

```
from hello import print_hello

print_hello("Hello, world")
```


# Examples

The tutorial contains a few examples that illustate key Legate concepts:

1. [Hello World](examples/hello-world.md): Shows the basics of creating tasks and adding task arguments.
1. [Variance](examples/variance.md): Shows how to create input arrays and tasks operating on partitioned data.
Also shows how to perform reduction tasks like summation.



