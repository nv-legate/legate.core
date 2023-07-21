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

# Basic build

If you are building on a cluster, first check if there are specialized scripts
available for your cluster at
[nv-legate/quickstart](https://github.com/nv-legate/quickstart). Even if your
specific cluster is not covered, you may be able to adapt an existing workflow.

## Getting dependencies through conda

The primary method of retrieving dependencies for Legate Core and downstream
libraries is through [conda](https://docs.conda.io/en/latest/). You will need
an installation of conda to follow the instructions below.

Please use the `scripts/generate-conda-envs.py` script to create a conda
environment file listing all the packages that are required to build, run and
test Legate Core and all downstream libraries. For example:

```shell
$ ./scripts/generate-conda-envs.py --python 3.10 --ctk 11.7 --os linux --compilers --openmpi
--- generating: environment-test-linux-py310-cuda-11.7-compilers-openmpi.yaml
```

Run this script with `-h` to see all available configuration options for the
generated environment file (e.g. all the supported Python versions). See the
[Dependencies](#dependency-listing) section for more details.

Once you have this environment file, you can install the required packages by
creating a new conda environment:

```shell
conda env create -n legate -f <env-file>.yaml
```

or by updating an existing environment:

```shell
conda env update -f <env-file>.yaml
```

## Building through install.py

The Legate Core repository comes with a helper `install.py` script in the
top-level directory, that will build the C++ parts of the library and install
the C++ and Python components under the currently active Python environment.

To add GPU support, use the `--cuda` flag:

```shell
./install.py --cuda
```

You can specify the CUDA toolkit directory and the CUDA architecture you want to
target using the `--with-cuda` and `--arch` flags, e.g.:

```shell
./install.py --cuda --with-cuda /usr/local/cuda/ --arch ampere
```

By default the script relies on CMake's auto-detection for these settings.
CMake will first search the currently active Python/conda environment
for dependencies, then any common system-wide installation directories (e.g.
`/usr/lib`). If a dependency cannot be found but is publicly available in source
form (e.g. OpenBLAS), cmake will fetch and build it automatically. You can
override this search by providing an install location for any dependency
explicitly, using a `--with-<dep>` flag, e.g. `--with-nccl` and
`--with-openblas`.

For multi-node execution Legate can use [GASNet](https://gasnet.lbl.gov/) (use
`--network gasnet1` or `--network gasnetex`) or [UCX](https://openucx.org) (use
`--network ucx`).
With gasnet1 or gasnetex, GASNet will be automatically downloaded and built,
but if you have an existing installation then you can inform the install script
using the `--with-gasnet` flag. You also need to specify the interconnect network
of the target machine using the `--conduit` flag.
With UCX, the library must be already installed and `--with-ucx` can be used
to point to the installation path if UCX is not installed under common system paths.
At least version 1.14 is required, configured with `--enable-mt`.

Compiling with networking support requires MPI.

For example this would be an installation for a
[DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/):

```shell
./install.py --network gasnet1 --conduit ibv --cuda --arch ampere
```

Alternatively, here is an install line for the
[Piz-Daint](https://www.cscs.ch/computers/decommissioned/piz-daint-piz-dora/)
supercomputer:

```shell
./install.py --network gasnet1 --conduit aries --cuda --arch pascal
```

To see all available configuration options, run with the `--help` flag:

```shell
./install.py --help
```

# Advanced topics

## Dependency listing

### Operating system

Legate has been tested on Linux and MacOS, although only a few flavors of Linux
such as Ubuntu have been thoroughly tested. There is currently no support for
Windows.

Specify your OS when creating a conda environment file through the `--os` flag
of `generate-conda-envs.py`.

### Python >= 3.9

In terms of Python compatibility, Legate *roughly* follows the timeline outlined
in [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html).

Specify your desired Python version when creating a conda environment file
through the `--python` flag of `generate-conda-envs.py`.

### C++17 compatible compiler

For example: g++, clang, or nvc++.

If you want to pull the compilers from conda, use an environment file created by
`generate-conda-envs.py` using the `--compilers` flag. An appropriate compiler
for the target OS will be chosen automatically.

If you need/prefer to use the system-provided compilers (typical for HPC
installations), please use a conda environment generated with `--no-compilers`.
Note that this will likely result in a
[conda/system library conflict](#alternative-sources-for-dependencies),
since the system compilers will typically produce executables
that link against the system-provided libraries, which can shadow the
conda-provided equivalents.

### CUDA >= 10.2 (optional)

Only necessary if you wish to run with Nvidia GPUs.

Some CUDA components necessary for building, e.g. the `nvcc` compiler and driver
stubs, are not distributed through conda. These must instead be installed using
[system-level packages](https://developer.nvidia.com/cuda-downloads). If these
are not installed under a standard system location, you will need to inform
`install.py` of their location using `--with-cuda`.

If you intend to pull any CUDA libraries from conda (see below), conda will need
to install an environment-local copy of the CUDA toolkit, even if you have it
installed system-wide. To avoid versioning conflicts it is safest to match the
version of CUDA installed system-wide, by specifying it to
`generate-conda-envs.py` through the `--ctk` flag.

Legate is tested and guaranteed to be compatible with Volta and later GPU
architectures. You can use Legate with Pascal GPUs as well, but there could
be issues due to lack of independent thread scheduling. Please report any such
issues on GitHub.

### CUDA Libraries (optional)

Only necessary if you wish to run with Nvidia GPUs.

The following additional CUDA libraries are required:

- `curand` (only necessary to provide this if building without CUDA support;
  CUDA-enabled installations will use the version bundled with CUDA)
- `cutensor` >= 1.3.3 (included in conda environment file)
- `nccl` (included in conda environment file)
- `thrust` >= 1.15 (pulled from github)

If you wish to provide alternative installations for these, then you can remove
them from the environment file (or invoke `generate-conda-envs.py` with `--ctk
none`, which will skip them all), and pass the corresponding `--with-<dep>` flag
to `install.py` (or let the build process attempt to locate them automatically).

### Build tools

The following tools are used for building Legate, and are automatically included
in the environment file:

- `cmake`
- `git`
- `make`
- `ninja` (this is optional, but produces more informative build output)
- `rust`
- `scikit-build`

### OpenBLAS

This library is automatically pulled from conda. If you wish to provide an
alternative installation, then you can manually remove `openblas` from the
generated environment file and pass `--with-openblas` to `install.py`.

Note that if you want to build OpenBLAS from source you will need to get a
Fortran compiler, e.g. by pulling `fortran-compiler` from conda-forge.

If you wish to compile Legate with OpenMP support, then you need a build of
OpenBLAS configured with the following options:

- `USE_THREAD=1`
- `USE_OPENMP=1`
- `NUM_PARALLEL=32` (or at least as many as the NUMA domains on the target
  machine) -- The `NUM_PARALLEL` flag defines how many instances of OpenBLAS's
  calculation API can run in parallel. Legate will typically instantiate a
  separate OpenMP group per NUMA domain, and each group can launch independent
  BLAS work. If `NUM_PARALLEL` is not high enough, some of this parallel work
  will be serialized.

### Numactl (optional)

Required to support CPU and memory binding in the Legate launcher.

Not available on conda; typically available through the system-level package
manager.

### MPI (optional)

Only necessary if you wish to run on multiple nodes.

Environments created using the `--openmpi` flag of `generate-conda-envs.py` will
contain the (generic) build of OpenMPI that is available on conda-forge. You may
need/prefer to use a more specialized build, e.g. the one distributed by
[MOFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/),
or one provided by your HPC vendor. In that case you should use an environment
file generated with `--no-openmpi`.

Legate requires a build of MPI that supports `MPI_THREAD_MULTIPLE`.

### Infiniband/RoCE networking libraries (optional)

Only necessary if you wish to run on multiple nodes, using the corresponding
networking hardware.

Not available on conda; typically available through MOFED or the system-level
package manager.

### UCX >= 1.14

Only necessary if you wish to run on multiple nodes, using the UCX Realm
networking backend.

You can use the version of UCX available on conda-forge by using an environment
file created by `generate-conda-envs.py` using the `--ucx` flag. Note that this
build of UCX might not include support for the particular networking hardware on
your machine (or may not be optimally tuned for such). In that case you may want
to use an environment file generated with `--no-ucx`, get UCX from another
source (e.g. MOFED, the system-level package manager, or compiled manually from
[source](https://github.com/openucx/ucx)), and pass the location of your
installation to `install.py` (if necessary) using `--with-ucx`.

Legate requires a build of UCX configured with `--enable-mt`.

## Alternative sources for dependencies

If you do not wish to use conda for some (or all) of the dependencies, you can
remove the corresponding entries from the environment file before passing it to
conda. See [the `install.py` section](#building-through-installpy) for
instructions on how to provide alternative locations for these dependencies to
the build process.

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

```shell
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

This way you can make sure that the (typically more recent) conda version of any
common library will be preferred over the system-wide one, no matter which
component requests it first.

## Building through pip

Legate Core is not yet registered in a standard pip repository. However, users
can still use the pip installer to build and install Legate Core. The following
command will trigger a single-node, CPU-only build of Legate Core, then install
it into the currently active Python environment:

```shell
$ pip install .
```

or

```shell
$ python3 -m pip install .
```

Legate relies on CMake to select its toolchain and build flags. Users can set
the environment variables `CXX` or `CXXFLAGS` prior to building to override the
CMake defaults.

Alternatively, CMake and build tool arguments can be passed via the
`CMAKE_ARGS`/`SKBUILD_CONFIGURE_OPTIONS` and `SKBUILD_BUILD_OPTIONS`
[environment variables](https://scikit-build.readthedocs.io/en/latest/usage.html#environment-variable-configuration):

```shell
$ CMAKE_ARGS="${CMAKE_ARGS:-} -D Legion_USE_CUDA:BOOL=ON" \
  pip install .
```

An alternative syntax using `setup.py` with `scikit-build` is

```shell
$ python setup.py install -- -DLegion_USE_CUDA:BOOL=ON
```

## Building through pip & cmake

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

There are several examples in the `scripts` folder. We walk through the steps in
`build-separately-no-install.sh` here.

First, the CMake build needs to be configured:

```shell
$ cmake -S . -B build -GNinja -D Legion_USE_CUDA=ON
```

Once configured, we can build the C++ libraries:

```shell
$ cmake --build build
```

This will invoke Ninja (or make) to execute the build.
Once the C++ libraries are available, we can do an editable (development) pip installation.

```shell
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
