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
an installation of conda to follow the instructions below. We suggest using
the [mamba](https://github.com/mamba-org/mamba) implementation of the conda
package manager.

Please use the `scripts/generate-conda-envs.py` script to create a conda
environment file listing all the packages that are required to build, run and
test Legate Core and all downstream libraries. For example:

```shell
$ ./scripts/generate-conda-envs.py --python 3.10 --ctk 12.2.2 --os linux --ucx
--- generating: environment-test-linux-py310-cuda-12.2.2-ucx.yaml
```

Run this script with `-h` to see all available configuration options for the
generated environment file (e.g. all the supported Python versions). See the
[Dependencies](#dependency-listing) section for more details.

Once you have this environment file, you can install the required packages by
creating a new conda environment:

```shell
mamba env create -n legate -f <env-file>.yaml
```

or by updating an existing environment:

```shell
mamba env update -f <env-file>.yaml
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
`--network gasnet1` or `--network gasnetex`, see [below](#gasnet-optional) for
more details) or [UCX](https://openucx.org) (use `--network ucx`, see
[below](#ucx-optional) for more details).

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

## Support matrix

The following table lists Legate's minimum supported versions of major dependencies.

"Full support" means that the corresponding versions (and all later ones) are
being tested with some regularity, and are expected to work. Please report any
incompatibility you find against a fully-supported version by opening a bug.

"Best-effort support" means that the corresponding versions are not actively
tested, but Legate should be compatible with them. We will not actively work to
fix any incompatibilities discovered under these versions, but we accept
contributions that fix such incompatibilities.

| Dependency       | Full support (min version)      | Best-effort support (min version)    |
| ---------------- | ------------------------------- | ------------------------------------ |
| CPU architecture | x86-64 (Haswell), aarch64       | ppc64le, older x86-64, Apple Silicon |
| OS               | RHEL 8, Ubuntu 20.04, MacOS 12  | other Linux                          |
| C++ compiler     | gcc 8, clang 7, nvc++ 19.1      | any compiler with C++17 support      |
| GPU architecture | Volta                           | Pascal                               |
| CUDA toolkit     | 11.4                            | 10.0                                 |
| Python           | 3.9                             |                                      |
| NumPy            | 1.22                            |                                      |

## Dependency listing

In this section we comment further on our major dependencies. Please consult an
environment file created by `generate-conda-envs.py` for a full listing of
dependencies, e.g. building and testing tools, and for exact version
requirements.

### Operating system

Legate has been tested on Linux and MacOS, although only a few flavors of Linux
such as Ubuntu have been thoroughly tested. There is currently no support for
Windows.

Specify your OS when creating a conda environment file through the `--os` flag
of `generate-conda-envs.py`.

### Python

In terms of Python compatibility, Legate *roughly* follows the timeline outlined
in [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html).

Specify your desired Python version when creating a conda environment file
through the `--python` flag of `generate-conda-envs.py`.

### C++ compiler

We suggest that you avoid using the compiler packages available on conda-forge.
These compilers are configured with the specific goal of building
redistributable conda packages (e.g. they explicitly avoid linking to system
directories), which tends to cause issues for development builds. Instead prefer
the compilers available from your distribution's package manager (e.g. apt/yum)
or your HPC vendor.

If you want to pull the compilers from conda, use an environment file created
by `generate-conda-envs.py` using the `--compilers` flag. An appropriate
compiler for the target OS will be chosen automatically.

### CUDA (optional)

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

### CUDA libraries (optional)

Only necessary if you wish to run with Nvidia GPUs.

The following additional CUDA libraries are required, for use by legate.core or
downstream libraries. Unless noted otherwise, these are included in the conda
environment file.

- `cublas`
- `cufft`
- `curand` (can optionally be used for its host fallback implementations even
  when building without CUDA support)
- `cusolver`
- `cutensor`
- `nccl`
- `nvml`
- `nvtx`
- `thrust` (pulled from github)

If you wish to provide alternative installations for these, then you can remove
them from the environment file (or invoke `generate-conda-envs.py` with `--ctk
none`, which will skip them all), and pass the corresponding `--with-<dep>` flag
to `install.py` (or let the build process attempt to locate them automatically).

### OpenBLAS

Used by cuNumeric for implementing linear algebra routines on CPUs.

This library is automatically pulled from conda. If you wish to provide an
alternative installation, then you can manually remove `openblas` from the
generated environment file and pass `--with-openblas` to cuNumeric's
`install.py`.

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

### TBLIS

Used by cuNumeric for implementing tensor contraction routines on CPUs.

This library will be automatically downloaded and built during cuNumeric
installation. If you wish to provide an alternative installation, pass
`--with-tblis` to cuNumeric's `install.py`.

cuNumeric requires a build of TBLIS configured as follows:

```
--with-label-type=int32_t --with-length-type=int64_t --with-stride-type=int64_t
```

and additionally `--enable-thread-model=openmp` if cuNumeric is compiled
with OpenMP support.

### Numactl (optional)

Required to support CPU and memory binding in the Legate launcher.

Not available on conda; typically available through the system-level package
manager.

### MPI (optional)

Only necessary if you wish to run on multiple nodes.

We suggest that you avoid using the generic build of OpenMPI available on
conda-forge. Instead prefer an MPI installation provided by your HPC vendor, or
from system-wide distribution channels like apt/yum and
[MOFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/),
since these will likely be more compatible with (and tuned for) your particular
system.

If you want to use the OpenMPI distributed on conda-forge, use an environment
file created by `generate-conda-envs.py` using the `--openmpi` flag.

Legate requires a build of MPI that supports `MPI_THREAD_MULTIPLE`.

### RDMA/networking libraries (e.g. Infiniband, RoCE, Slingshot)  (optional)

Only necessary if you wish to run on multiple nodes, using the corresponding
networking hardware.

Not available on conda; typically available through MOFED or the system-level
package manager.

Depending on your hardware, you may need to use a particular Realm
networking backend, e.g. as of October 2023 HPE Slingshot is only
compatible with GASNet.

### GASNet (optional)

Only necessary if you wish to run on multiple nodes, using the GASNet1 or
GASNetEx Realm networking backend.

This library will be automatically downloaded and built during Legate
installation. If you wish to provide an alternative installation, pass
`--with-gasnet` to `install.py`.

When using GASNet, you also need to specify the interconnect network of the
target machine using the `--conduit` flag.

### UCX (optional)

Only necessary if you wish to run on multiple nodes, using the UCX Realm
networking backend.

You can use the version of UCX available on conda-forge by using an environment
file created by `generate-conda-envs.py` using the `--ucx` flag. Note that this
build of UCX might not include support for the particular networking hardware on
your machine (or may not be optimally tuned for such). In that case you may want
to use an environment file generated with `--no-ucx` (default), get UCX from
another source (e.g. MOFED, the system-level package manager, or compiled
manually from [source](https://github.com/openucx/ucx)), and pass the location
of your UCX installation to `install.py` (if necessary) using `--with-ucx`.

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
