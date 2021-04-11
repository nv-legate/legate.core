<!--
Copyright 2021 NVIDIA Corporation

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

# Legate 

Legate is an endeavor to democratize computing by making it possible
for all programmers to leverage the power of large clusters of GPUs
with minimal effort. Legate aims to achieve this by delivering two 
different, but related, contributions:

1. Legate provides drop-in replacements of popular Python libraries
   that are capable of running on machines of any scale so that users
   never need to change their code when scaling from a laptop/desktop 
   to a supercomputer or the cloud.
2. Legate provides composability of libraries through a common data
   model and implicit extraction of parallelism so that Python programs
   constructed by combining multiple Legate libraries will incur
   near-zero overhead in passing distributed data types between libraries, 
   and sufficient parallelism will be extracted across abstraction boundaries
   to run at the speed of light on the target machine.

In visual terms:

![](vision.png)

1. [Why Legate?](#why-legate)
2. [How Do I Install Legate?](#how-do-i-install-legate)
3. [How Do I Use Legate?](#how-do-i-use-legate)
4. [Supported and Planned Features](#supported-and-planned-features)
5. [How Does Legate Work?](#how-does-legate-work)
6. [Next Steps](#next-steps)

## Why Legate?

Programming large clusters of GPUs shouldn't be so hard. Today more and
more data scientists and HPC programmers are forced to use large clusters of
GPUs to handle the size of their data and computational intensity of
their workloads. However, this class of users often struggle to program large 
machines as they must learn to use both multi-node programming frameworks
like Spark or MPI, as well as GPU programming models such as CUDA or
OpenACC. For users that primarily develop in programming models with a
single thread of control and a shared memory abstraction such as in Python 
learning and combining multiple programming models with explicit parallelism
and distinct address spaces is a daunting proposition. Too often they will
abandon their efforts and unfortunately limit themselves to problems that can 
be solved only on their desktop machine.

Legate aims to give these programmers the best of both worlds. Users
can develop their code using standard Python libraries and test it using 
small problem sizes on their desktop machine. Later they can deploy the 
exact same code using Legate across a large cluster of GPUs on a much 
larger data set. Legate should greatly expand the set of users that can 
leverage the computational power and potential of large clusters of GPUs.

## How Do I Install Legate?

### Dependencies

Legate has been tested on Ubuntu 16.04 and 18.04.  We expect it to be possible 
to install legate in other operating systems, but such installations are not tested.

We currently assume the following existing dependencies are already installed:

* Python 3.5 or later
* Python packages listed in the `requirements.txt` file
* [CUDA](https://developer.nvidia.com/cuda-downloads) - version 8.0 or later
* [MPI](https://www.open-mpi.org/) - for multi-node versions only (we really only need the MPI compiler and the PMI library to bootstrap GASNet, but it's easier to have a full MPI installation)

Legate comes with a `requirements.txt` file, which can be used to conveniently 
install Python dependencies:

```
pip install -r requirements.txt
```

### Installation

The Legate Core library comes with both a standard `setup.py` script and a 
custom `install.py` script in the top-level directory of the repository that 
will build and install the Legate Core library. Users can use either script
to install Legate as they will produce the same effect. A simple 
single-node, CPU-only installation of Legate into `targetdir` will be performed by:
```
./setup.py --install-dir targetdir
```
To add GPU support for Legate simply use the `--cuda` flag. The first time you request
GPU support you will need to use the `--with-cuda` flag to specify the location
of your CUDA installation. For later invocations you do not need to use this 
flag again the installation scripts will remember the location of your CUDA 
installation until you tell it differently. You can also specify the name of the
CUDA architecture you want to target with the `--arch` flag (the default is `volta`
but you can also specify `kepler`, `maxwell`, or `pascal`).
```
./install.py --cuda --with-cuda /usr/local/cuda/ --arch volta
```
For multi-node support Legate uses [GASNet](https://gasnet.lbl.gov/) which can be
requested using the the `--gasnet` flag. If you have an existing GASNet installation
then you can inform the install script with the `--with-gasnet` flag. The 
`install.py` script also requires you to specify the interconnect network of
the target machine using the `--conduit` flag (current choices are one of `ibv`
for [Infiniband](http://www.infinibandta.org/), or `gemini` or `aries` for Cray 
interconnects). For example this would be an installation for a cluster of 
DGX1-V boxes like [SaturnV](https://www.nvidia.com/en-us/data-center/dgx-saturnv/):
```
./install.py --gasnet --conduit ibv --cuda --arch volta
```
Alternatively here is an install line for the 
[Piz-Daint](https://www.cscs.ch/computers/dismissed/piz-daint-piz-dora/) supercomputer:
```
./install.py --gasnet --conduit aries --cuda --arch pascal
```
To see all the options available for installing Legate, just run with the `--help` flag:
```
./install.py --help
```
Options passed to `setup.py` will automatically be forwarded to `install.py` so
that users can use them interchangeably (this provides backwards compatibility
for earlier versions of Legate when only `install.py` existed).

### Python used by Legate

Legate discovers the Python library and version during build time, and then it 
builds all successive Legate libraries against that version of Python. The build system 
tries to detect the Python setup from the default Python interpreter, but sometimes 
it is unsuccessful or a different version of Python than the one in the environment 
may be desired. To use a different version of Python than the one available in the 
environment, the `PYTHON_ROOT` variable must be set to the base directory of the 
desired Python installation.

Sometimes, the search for the Python library may fail.  In such situation, the 
build system generates a warning:
```
runtime.mk:265: cannot find libpython3.6*.so - falling back to using LD_LIBRARY_PATH
```
In this case, Legate will use the Python library that is available at runtime, if any.
To explicitly specify the Python library to use, `PYTHON_LIB` should be set to the 
location of the library, and `PYTHON_VERSION_MAJOR` should be set to the `2` or `3` 
as appropriate.

### Toolchain selection

Legate relies on environment variables to select its toolchain and build flags 
(such as `CXX`, `CC_FLAGS`, `LD_FLAGS`, `NVCC_FLAGS`). Setting these environment
variables prior to building and installing Legate will influence the build of 
any C++ and CUDA code in Legate.

## How Do I Use Legate?

After installing the Legate Core library, the next step is to install a Legate
application library such as Legate NumPy. The installation process for a 
Legate application library will require you to provide a pointer to the location
of your Legate Core library installation as this will be used to configure the
installation of the Legate application library. After you finish installing any
Legate application libraries, you can then simply replace their `import` statements
with the equivalent ones from any Legate application libraries you have installed.
For example, you can change this:
```python
import numpy as np
```
to this:
```python
import legate.numpy as np
```
After this, you can use the `legate` driver script in the `bin` directory of
your installation to run any Python program. For example, to run your script 
in the default configuration (4 CPUs cores and 4 GB of memory) just run:
```
installdir/bin/legate my_python_program.py [other args]
```
The `legate` script also allows you to control the amount of resources that 
Legate consumes when running on the machine. The `--cpus` and `--gpus` flags
are used to specify how many CPU and GPU processors should be used on a node.
The `--sysmem` flag can be used to specify how many MBs of DRAM Legate is allowed
to use per node, while the `--fbmem` flag controls how many MBs of framebuffer 
memory Legate is allowed to use per GPU. For example, when running on a DGX 
station, you might run your application as follows:
```
installdir/bin/legate --cpus 16 --gpus 4 --sysmem 100000 -fbmem 15000 my_python_program.py
```
This will make 16 CPU processors and all 4 GPUs available for use by Legate.
It will also allow Legate to consume up to 100 GB of DRAM memory and 15 GB of
framebuffer memory per GPU for a total of 60 GB of GPU framebuffer memory. Note
that you probably will not be able to make all the resources of the machine
available for Legate as some will be used by the system or Legate itself for 
meta-work. Currently if you try to exceed these resources during execution then
Legate will inform you that it had insufficient resources to complete the job
given its current mapping heuristics. If you believe the job should fit within
the assigned resources please let us know so we can improve our mapping heuristics.
There are many other flags available for use in the `legate` driver script that you
can use to communicate how Legate should view the available machine resources.
You can see a list of them by running:
```
installdir/bin/legate --help
```
In addition to running NumPy programs, you can also use Legate in an interactive
mode by simply not passing any `*py` files on the command line. You can still
request resources just as you would though with a normal file. Legate will 
still use all the resources available to it, including doing multi-node execution.
```
installdir/bin/legate --cpus 16 --gpus 4 --sysmem 100000 -fbmem 15000
Welcome to Legion Python interactive console
>>>
```
Note that Legate does not currently support multi-tenancy cases where different
users are attempting to use the same hardware concurrently.

### Distributed Launch

If legate is compiled with GASNet support ([see the installation section](#Installation)), 
it can be run in parallel by using the `--nodes` option followed by the number of nodes 
to be used.  Whenever the `--nodes` option is used, Legate will be launched 
using `mpirun`, even with `--nodes 1`.  Without the `--nodes` option, no launcher will 
be used.  Legate currently supports `mpirun` and `jsrun` as launchers and we are considering
adding additional launcher kinds if you would like to advocate for one. You can select the
target kind of launcher with `--launcher`.

### Debugging and Profiling

Legate also comes with several tools that you can use to better understand
your program both from a correctness and a performance standpoint. For 
correctness, Legate has facilities for constructing both dataflow
and event graphs for the actual run of an application. These graphs require
that you have an installation of [GraphViz](https://www.graphviz.org/)
available on your machine. To generate a dataflow graph for your Legate
program simply pass the `--dataflow` flag to the `legate.py` script and after
your run is complete we will generate a `dataflow_legate.pdf` file containing
the dataflow graph of your program. To generate a corresponding event graph
you simply need to pass the `--event` flag to the `legate.py` script to generate
a `event_graph_legate.pdf` file. These files can grow to be fairly large for non-trivial 
programs so we encourage you to keep your programs small when using these 
visualizations or invest in a [robust PDF viewer](https://get.adobe.com/reader/).

For profiling all you need to do is pass the `--profile` flag to Legate and
afterwards you will have a `legate_prof` directory containing a web page that
can be viewed in any web browser that displays a timeline of your program's
execution. You simply need to load the `index.html` page from a browser. You 
may have to enable local JavaScript execution if you are viewing the page from
your local machine (depending on your browser).

We recommend that you do not mix debugging and profiling in the same run as
some of the logging for the debugging features requires significant file I/O 
that can adversely effect the performance of the application.

## How Does Legate Work?

Legate is built on top of the [Legion](http://legion.stanford.edu) 
programming model and runtime system. Legion is primarily designed for large
HPC applications that target supercomputers and consequently applications written
in the Legion programming model tend to both perform and scale well on large
clusters of both CPUs and GPUs. Legion programs also are easy to port to
new machines as they inherently decouple the machine-independent 
specification from decisions about how that application is mapped to the
target machine. Due to this more abstract nature, many programmers find
writing Legion programs challenging. Legate solves the problem of translating
NumPy programs to the Legion programming model so that users can have the
best of both worlds: high productivity using a sequential programming 
environment they know (Python) while still reaping the performance and 
scalability benefits of Legion when running on large machines.

Every Legate application library translates their components of programs
down to Legion. Data types from libraries, such as arrays in Legate NumPy
are mapped down to Legion data types such as logical regions or futures.
In the case of regions, Legate application libraries rely heavily on 
Legion's support for partitioning of logical regions into subregions.
Each library has its own heuristics for computing such partitions that
take into consideration the computations that will access the data, the 
ideal sizes of data to be consumed by different processor kinds, and
the available number of processors. Computations in Legate application
libraries are converted in Legion tasks. Each Legate application library
comes with its own custom mapper that uses heuristics to determine the best 
choice of mapping for tasks (e.g. are they best run on a CPU or a GPU). All 
Legate tasks are implemented in native C or CUDA in order to achieve excellent 
performance on the target processor kind. Importantly, by using Legion, 
Legate is able to control the placement of data in order to leave it 
in-place in fast memories like GPU framebuffers across NumPy operations. 

When running on large clusters, Legate leverages a novel technology provided
by Legion called "control replication" to avoid the sequential bottleneck
of having one node farm out work to all the nodes in the cluster. With
control replication, Legate will actually replicate the Python program and
run it across all the nodes of the machine at the same time. As each copy
of the NumPy program runs and launches off NumPy operations, Legate 
automatically translates these to Legion "index space" tasks with each
"point" task operating on different subregions of the data. With control
replication, each node becomes responsible for a subset of the points in
the index space task launch. When communication is necessary between 
NumPy operations, the Legion runtime's program analysis will automatically
detect it and insert the necessary data movement and synchronization 
across nodes (or GPU framebuffers). This is the secret to allowing
sequential NumPy programs to run efficiently at scale across large clusters
of GPUs and operate on huge data sets that could never fit in memory
in a single workstation.

## Next Steps

We recommend starting by experimenting with at least one Legate application
library to test out performance and see how Legate works. If you are interested
in building your own Legate application library, we recommend that you 
investigate our Legate Hello World application library that provides a small
example of how to get started developing your own drop-in replacement library
on top of Legion using the Legate Core library.

