<!--
Copyright 2021-2022 NVIDIA Corporation

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

# TL;DR

1) Check if there are specialized scripts available for your cluster at https://github.com/nv-legate/quickstart.
2) [Install dependencies from conda](#getting-dependencies-through-conda)
3) [Build using install.py](#using-installpy)

# Getting dependencies

## Getting dependencies through conda

The primary method of retrieving dependencies for Legate Core and downstream
libraries is through [conda](https://conda.io). You will need an installation of
conda to follow the instructions below.

Please use the `scripts/generate-conda-envs.py` script to create a conda
environment file listing all the packages that are required to build, run and
test Legate Core and all downstream libraries. For example:

```
$ ./scripts/generate-conda-envs.py --python 3.10 --ctk 11.7 --os linux --compilers --openmpi
--- generating: environment-test-linux-py310-cuda-11.7-compilers-openmpi.yaml
```

Run this script with `-h` to see all available configuration options for the
generated environment file (e.g. all the supported Python versions). See the
[Notable Dependencies](#notable-dependencies) section for more details.

Once you have this environment file, you can install the required packages by
creating a new conda environment:

```
conda env create -n legate -f <env-file>.yaml
```

or by updating an existing environment:

```
conda env update -f <env-file>.yaml
```

## Notable dependencies

### OS (`--os` option)

Legate has been tested on Linux and MacOS, although only a few flavors of Linux
such as Ubuntu have been thoroughly tested. There is currently no support for
Windows.

### Python >= 3.8 (`--python` option)

In terms of Python compatibility, Legate *roughly* follows the timeline outlined
in [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html).

### C++17 compatible compiler (`--compilers` option)

For example: g++, clang, or nvc++. When creating an environment using the
`--compilers` flag, an appropriate compiler for the current system will be
pulled from conda.

If you need/prefer to use the system-provided compilers (typical for HPC
installations), please use a conda environment generated with `--no-compilers`.
Note that this will likely result in a
[conda/system library conflict](#alternative-sources-for-dependencies),
since the system compilers will typically produce executables
that link against the system-provided libraries, which can shadow the
conda-provided equivalents.

### CUDA >= 10.2 (`--ctk` flag; optional)

Only necessary if you wish to run with Nvidia GPUs.

Some CUDA components necessary for building, e.g. the `nvcc` compiler and driver
stubs, are not distributed through conda. These must instead be installed using
[system-level packages](https://developer.nvidia.com/cuda-downloads).

Independent of the system-level CUDA installation, conda will need to install an
environment-local copy of the CUDA toolkit (which is what the `--ctk` option
controls). To avoid versioning conflicts it is safest to match the version of
CUDA installed system-wide on your machine

Legate is tested and guaranteed to be compatible with Volta and later GPU
architectures. You can use Legate with Pascal GPUs as well, but there could
be issues due to lack of independent thread scheduling. Please report any such
issues on GitHub.

### Fortran compiler (optional)

Only necessary if you wish to build OpenBLAS from source.

Not included by default in the generated conda environment files; install
`fortran-compiler` from `conda-forge` if you need it.

### Numactl (optional)

Required to support CPU and memory binding in the Legate launcher.

Not available on conda; typically available through the system-level package
manager.

### MPI (`--openmpi` option; optional)

Only necessary if you wish to run on multiple nodes.

Conda distributes a generic build of OpenMPI, but you may need to use a more
specialized build, e.g. the one distributed by
[MOFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/),
or one provided by your HPC vendor. In that case you should use an environment
file generated with `--no-openmpi`.

Legate requires a build of MPI that supports `MPI_THREAD_MULTIPLE`.

### Networking libraries (e.g. Infiniband, RoCE, UCX; optional)

Only necessary if you wish to run on multiple nodes.

Not available on conda; typically available through MOFED or the system-level
package manager.

If using UCX, a build configured with `--enable-mt` is required.

## Alternative sources for dependencies

If you do not wish to use conda for some (or all) of the dependencies, you can
remove the corresponding entries from the environment file before passing it to
conda. See [the `install.py` section](#using-installpy) for instructions on how
to provide alternative locations for these dependencies to the build process.

Note that this is likely to result in conflicts between conda-provided and
system-provided libraries.

Conda distributes its own version of certain common libraries (in particular the
C++ standard library), which are also typically available system-wide. Any
system package you include will typically link to the system version, while
conda packages link to the conda version. Often these two different versions,
although incompatible, carry the same version number (`SONAME`), and are
therefore indistinguishable to the dynamic linker. Then, the first component to
specify a link location for this library will cause it to be loaded from there,
and any subsequent link requests for the same library, even if suggesting a
different link location, will get served using the previously linked version.

This can cause link failures at runtime, e.g. when a system-level library
happens to be the first to load GLIBC, causing any conda library that comes
after to trip GLIBC's internal version checks, since the conda library expects
to find symbols with more recent version numbers than what is available on the
system-wide GLIBC:

```
/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /opt/conda/envs/legate/lib/libarrow.so)
```

You can usually work around this issue by putting the conda library directory
first in the dynamic library resolution path:

```
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

This way you can make sure that the (typically more recent) conda version of any
common library will be preferred over the system-wide one, no matter which
component requests it first.

# Building for Users

## Using install.py

The Legate Core repository comes with a helper `install.py` script in the
top-level directory, that will build the C++ parts of the library and install
the C++ and Python components under the currently active Python environment.

To add GPU support, use the `--cuda` flag:

```
./install.py --cuda
```

You can specify the CUDA toolkit directory and the CUDA architecture you want to
target using the `--with-cuda` and `--arch` flags, e.g.:

```
./install.py --cuda --with-cuda /usr/local/cuda/ --arch ampere
```

By default the script relies on CMake's auto-detection for these settings.
CMake will first search the currently active Python/conda environment
for dependencies, then any common system-wide installation directories (e.g.
`/usr/lib`). If a dependency cannot be found but is publicly available in source
form (e.g. OpenBLAS), cmake will fetch and build it automatically. You can
override this search by providing an install location for any dependency
explicitly, using a `--with-dep` flag, e.g. `--with-nccl` and
`--with-openblas`.

For multi-node execution Legate uses [GASNet](https://gasnet.lbl.gov/) which can be
requested using the `--network gasnet1` or `--network gasnetex` flag. By default
GASNet will be automatically downloaded and built, but if you have an existing
installation then you can inform the install script using the `--with-gasnet` flag.
You also need to specify the interconnect network of the target machine using the
`--conduit` flag.

For example this would be an installation for a
[DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/):
```
./install.py --network gasnet1 --conduit ibv --cuda --arch ampere
```
Alternatively, here is an install line for the
[Piz-Daint](https://www.cscs.ch/computers/dismissed/piz-daint-piz-dora/) supercomputer:
```
./install.py --network gasnet1 --conduit aries --cuda --arch pascal
```

To see all available configuration options, run with the `--help` flag:

```
./install.py --help
```

## Using pip

Legate Core is not yet registered in a standard pip repository. However, users
can still use the pip installer to build and install Legate Core. The following
command will trigger a single-node, CPU-only build of Legate Core, then install
it into the currently active Python environment:

```
$ pip install .
```
or
```
$ python3 -m pip install .
```

## Advanced Customization

Legate relies on CMake to select its toolchain and build flags. Users can set
the environment variables `CXX` or `CXXFLAGS` prior to building to override the
CMake defaults. Alternatively, CMake values can be overridden through the
`SKBUILD_CONFIGURE_OPTIONS` variable:

```
$ SKBUILD_CONFIGURE_OPTIONS="-D Legion_USE_CUDA:BOOL=ON" \
  pip install .
```

An alternative syntax using `setup.py` with `scikit-build` is

```
$ python setup.py install -- -DLegion_USE_CUDA:BOOL=ON
```

# Building for Developers

## Overview

pip uses [scikit-build](https://scikit-build.readthedocs.io/en/latest/)
in `setup.py` to drive the build and installation.  A `pip install` will trigger three general actions:

1. CMake build and installation of C++ libraries
2. CMake generation of configuration files and build-dependent Python files
3. pip installation of Python files

The CMake build can be configured independently of `pip`, allowing incremental C++ builds directly through CMake.
This simplifies rebuilding the C++ shared libraries either via command-line or via IDE.
After building the C++ libraries, the `pip install` can be done in "editable" mode using the `-e` flag.
This configures the Python site packages to import the Python source tree directly.
The Python source can then be edited and used directly for testing without requiring another `pip install`.

## Example

There are several examples in the `scripts` folder. We walk through the steps in
`build-separately-no-install.sh` here.

First, the CMake build needs to be configured:

```
$ cmake -S . -B build -GNinja -D Legion_USE_CUDA=ON
```

Once configured, we can build the C++ libraries:

```
$ cmake --build build
```

This will invoke Ninja (or make) to execute the build.
Once the C++ libraries are available, we can do an editable (development) pip installation.

```
$ SKBUILD_BUILD_OPTIONS="-D FIND_LEGATE_CORE_CPP=ON -D legate_core_ROOT=$(pwd)/build" \
  python3 -m pip install \
  --root / --no-deps --no-build-isolation
  --editable .
```

The Python source tree and CMake build tree are now available with the environment Python
for running Legate programs. The diagram below illustrates the
complete workflow for building both Legate core and a downstream package,
[cuNumeric](https://github.com/nv-legate/cunumeric)

<img src="docs/figures/developer-build.png" alt="drawing" width="600"/>
