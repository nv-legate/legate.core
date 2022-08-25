#!/usr/bin/env python

# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import platform
import shlex
import subprocess
import sys

os_name = platform.system()

if os_name == "Linux":
    dylib_ext = ".so"
    LIB_PATH = "LD_LIBRARY_PATH"
elif os_name == "Darwin":  # Don't currently support Darwin at the moment
    dylib_ext = ".dylib"
    LIB_PATH = "DYLD_LIBRARY_PATH"
else:
    raise Exception("Legate does not work on %s" % platform.system())


def read_c_define(header_path, def_name):
    try:
        with open(header_path, "r") as f:
            line = f.readline()
            while line:
                if line.startswith("#define"):
                    tokens = line.split(" ")
                    if tokens[1].strip() == def_name:
                        return tokens[2]
                line = f.readline()
        return None
    except IOError:
        return None


def read_cmake_var(pattern, filepath):
    return (
        subprocess.check_output(["grep", "--color=never", pattern, filepath])
        .decode("UTF-8")
        .strip()
        .split("=")[1]
    )


def get_python_site_packages_path(legion_dir):
    lib_dir = os.path.join(legion_dir, "lib")
    python_lib = None
    for f in os.listdir(lib_dir):
        if f.startswith("python") and not os.path.isfile(f):
            if python_lib is not None:
                print(
                    "WARNING: Found multiple python supports in your Legion "
                    "installation."
                )
                print("Using the following one: " + str(python_lib))
            else:
                python_lib = os.path.join(
                    lib_dir, os.path.join(f, "site-packages")
                )

    if python_lib is None:
        raise Exception("Cannot find a Legate python library")
    return python_lib


def get_legate_build_dir(legate_dir):
    join = os.path.join
    exists = os.path.exists
    # If using a local non-scikit-build CMake build dir, read
    # Legion_BINARY_DIR and Legion_SOURCE_DIR from CMakeCache.txt
    if exists(legate_build_dir := join(legate_dir, "build")) and exists(
        join(legate_build_dir, "CMakeCache.txt")
    ):
        pass
    elif exists(_skbuild_dir := join(legate_dir, "_skbuild")):
        for f in os.listdir(_skbuild_dir):
            # If using a local scikit-build dir at _skbuild/<arch>/cmake-build,
            # read Legion_BINARY_DIR and Legion_SOURCE_DIR from CMakeCache.txt
            if exists(
                legate_build_dir := join(_skbuild_dir, f, "cmake-build")
            ) and exists(join(legate_build_dir, "CMakeCache.txt")):
                cmake_cache_txt = join(legate_build_dir, "CMakeCache.txt")
                try:
                    # Test whether FIND_LEGATE_CORE_CPP is set to ON. If it
                    # isn't, then we built legate_core C++ as a side-effect of
                    # building legate_core_python.
                    read_cmake_var(
                        "FIND_LEGATE_CORE_CPP:BOOL=OFF", cmake_cache_txt
                    )
                except Exception:
                    # If FIND_LEGATE_CORE_CPP is set to ON, check to see if
                    # legate_core_DIR is a valid path. If it is, check whether
                    # legate_core_DIR is a path to a legate_core build dir i.e.
                    # `-D legate_core_ROOT=/legate.core/build`
                    legate_core_dir = read_cmake_var(
                        "legate_core_DIR:PATH=", cmake_cache_txt
                    )
                    # If legate_core_dir doesn't have a CMakeCache.txt, CMake's
                    # find_package found a system legate_core installation.
                    # Return the installation paths.
                    if os.path.exists(
                        cmake_cache_txt := join(
                            legate_core_dir, "CMakeCache.txt"
                        )
                    ):
                        return read_cmake_var(
                            "legate_core_BINARY_DIR:STATIC=", cmake_cache_txt
                        )
                    return None
                return legate_build_dir
            legate_build_dir = None
    else:
        legate_build_dir = None
    return legate_build_dir


def get_legate_paths():
    import legate

    join = os.path.join
    dirname = os.path.dirname

    legate_dir = dirname(legate.__path__[0])
    legate_build_dir = get_legate_build_dir(legate_dir)

    if legate_build_dir is None:
        return {
            "legate_dir": legate_dir,
            "legate_build_dir": legate_build_dir,
            "bind_sh_path": join(dirname(sys.argv[0]), "bind.sh"),
            "legate_lib_path": join(dirname(dirname(sys.argv[0])), "lib"),
        }

    cmake_cache_txt = join(legate_build_dir, "CMakeCache.txt")
    legate_source_dir = read_cmake_var(
        "legate_core_SOURCE_DIR:STATIC=", cmake_cache_txt
    )
    legate_binary_dir = read_cmake_var(
        "legate_core_BINARY_DIR:STATIC=", cmake_cache_txt
    )

    return {
        "legate_dir": legate_dir,
        "legate_build_dir": legate_build_dir,
        "bind_sh_path": join(legate_source_dir, "bind.sh"),
        "legate_lib_path": join(legate_binary_dir, "lib"),
    }


def get_legion_paths(legate_dir, legate_build_dir=None):

    #
    # Construct and return paths needed to launch `legion_python`,accounting
    # for multiple ways Legion and legate_core may be configured or installed.
    #
    # 1. Legion was found in a standard system location (/usr, $CONDA_PREFIX)
    # 2. Legion was built as a side-effect of building legate_core:
    #    ```
    #    SKBUILD_CONFIGURE_OPTIONS="" python -m pip install .
    #    ```
    # 3. Legion was built in a separate directory independent of legate_core
    #    and the path to its build directory was given when configuring
    #    legate_core:
    #    ```
    #    SKBUILD_CONFIGURE_OPTIONS="-D Legion_ROOT=/legion/build" \
    #        python -m pip install .
    #    ```
    #
    # Additionally, legate_core has multiple run modes:
    #
    # 1. As an installed Python module (`python -m pip install .`)
    # 2. As an "editable" install (`python -m pip install --editable .`)
    #
    # When determining locations of Legion and legate_core paths, prioritize
    # local builds over global installations. This allows devs to work in the
    # source tree and re-run without overwriting existing installations.
    #

    join = os.path.join
    dirname = os.path.dirname

    def installed_legion_paths(legion_dir, legion_module=None):
        return {
            "legion_bin_path": join(legion_dir, "bin"),
            "legion_lib_path": join(legion_dir, "lib"),
            "realm_defines_h": join(legion_dir, "include", "realm_defines.h"),
            "legion_defines_h": join(
                legion_dir, "include", "legion_defines.h"
            ),
            "legion_spy_py": join(legion_dir, "bin", "legion_spy.py"),
            "legion_prof_py": join(legion_dir, "bin", "legion_prof.py"),
            "legion_python": join(legion_dir, "bin", "legion_python"),
            "legion_module": legion_module,
        }

    if legate_build_dir is None:
        legate_build_dir = get_legate_build_dir(legate_dir)

    # If no local build dir found, assume legate installed into the python env
    if legate_build_dir is None:
        return installed_legion_paths(dirname(dirname(sys.argv[0])))

    # If a legate build dir was found, read `Legion_SOURCE_DIR` and
    # `Legion_BINARY_DIR` from in CMakeCache.txt, return paths into the source
    # and build dirs. This allows devs to quickly rebuild inplace and use the
    # most up-to-date versions without needing to install Legion and
    # legate_core globally.

    cmake_cache_txt = join(legate_build_dir, "CMakeCache.txt")

    try:
        # Test whether Legion_DIR is set. If it isn't, then we built Legion as
        # a side-effect of building legate_core
        read_cmake_var("Legion_DIR:PATH=Legion_DIR-NOTFOUND", cmake_cache_txt)
    except Exception:
        # If Legion_DIR is a valid path, check whether it's a
        # Legion build dir, i.e. `-D Legion_ROOT=/legion/build`
        legion_dir = read_cmake_var("Legion_DIR:PATH=", cmake_cache_txt)
        if os.path.exists(join(legion_dir, "CMakeCache.txt")):
            cmake_cache_txt = join(legion_dir, "CMakeCache.txt")
        # else:
        #     # If legion_dir doesn't have a CMakeCache.txt, CMake's find_package
        #     # found a system Legion installation. Return the installation paths.
        #     return installed_legion_paths(dirname(dirname(sys.argv[0])))

    try:
        # If Legion_SOURCE_DIR and Legion_BINARY_DIR are in CMakeCache.txt,
        # return the paths to Legion in the legate_core build dir.
        legion_source_dir = read_cmake_var(
            "Legion_SOURCE_DIR:STATIC=", cmake_cache_txt
        )
        legion_binary_dir = read_cmake_var(
            "Legion_BINARY_DIR:STATIC=", cmake_cache_txt
        )
        return {
            "legion_bin_path": join(legion_binary_dir, "bin"),
            "legion_lib_path": join(legion_binary_dir, "lib"),
            "realm_defines_h": join(
                legion_binary_dir, "runtime", "realm_defines.h"
            ),
            "legion_defines_h": join(
                legion_binary_dir, "runtime", "legion_defines.h"
            ),
            "legion_spy_py": join(legion_source_dir, "tools", "legion_spy.py"),
            "legion_prof_py": join(legion_source_dir, "tools", "legion_prof.py"),
            "legion_python": join(legion_binary_dir, "bin", "legion_python"),
            "legion_module": join(
                legion_source_dir, "bindings", "python", "build", "lib"
            ),
        }
    except Exception:
        pass

    # Otherwise return the installation paths.
    return installed_legion_paths(dirname(dirname(sys.argv[0])))


def run_legate(
    ranks,
    ranks_per_node,
    cpus,
    gpus,
    openmp,
    ompthreads,
    utility,
    sysmem,
    numamem,
    fbmem,
    zcmem,
    regmem,
    opts,
    profile,
    dataflow,
    event,
    log_dir,
    user_logging_levels,
    log_to_file,
    keep_logs,
    gdb,
    cuda_gdb,
    memcheck,
    module,
    nvprof,
    nsys,
    nsys_targets,
    nsys_extra,
    progress,
    freeze_on_error,
    mem_usage,
    not_control_replicable,
    launcher,
    verbose,
    gasnet_trace,
    eager_alloc,
    cpu_bind,
    mem_bind,
    gpu_bind,
    nic_bind,
    launcher_extra,
):
    if verbose:
        print("legate:          ", str(sys.argv[0]))
    # Build the environment for the subprocess invocation
    legate_paths = get_legate_paths()
    legate_dir = legate_paths["legate_dir"]
    bind_sh_path = legate_paths["bind_sh_path"]
    legate_lib_path = legate_paths["legate_lib_path"]
    legate_build_dir = legate_paths["legate_build_dir"]
    if verbose:
        print("legate_dir:      ", legate_dir)
        print("bind_sh_path:    ", bind_sh_path)
        print("legate_lib_path: ", legate_lib_path)
        print("legate_build_dir:", legate_build_dir)

    legion_paths = get_legion_paths(legate_dir, legate_build_dir)
    legion_lib_path = legion_paths["legion_lib_path"]
    realm_defines_h = legion_paths["realm_defines_h"]
    legion_defines_h = legion_paths["legion_defines_h"]
    legion_spy_py = legion_paths["legion_spy_py"]
    legion_prof_py = legion_paths["legion_prof_py"]
    legion_python = legion_paths["legion_python"]
    legion_module = legion_paths["legion_module"]
    if verbose:
        print("legion_lib_path: ", legion_lib_path)
        print("realm_defines_h: ", realm_defines_h)
        print("legion_defines_h:", legion_defines_h)
        print("legion_spy_py:   ", legion_spy_py)
        print("legion_prof_py:  ", legion_prof_py)
        print("legion_python:   ", legion_python)
        print("legion_module:   ", legion_module)

    cmd_env = dict(os.environ.items())
    # We never want to save python byte code for legate
    cmd_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Set the path to the Legate module as an environment variable
    # The current directory should be added to PYTHONPATH as well
    extra_python_paths = (
        cmd_env["PYTHONPATH"] if "PYTHONPATH" in cmd_env else []
    )
    if legion_module is not None:
        extra_python_paths.append(legion_module)
    # Make sure the base directory for this file is in the python path
    extra_python_paths.append(os.path.dirname(os.path.dirname(__file__)))
    cmd_env["PYTHONPATH"] = os.pathsep.join(extra_python_paths)

    # Make sure the version of Python used by Realm is the same as what the
    # user is using currently.
    curr_pyhome = os.path.dirname(os.path.dirname(sys.executable))
    realm_pylib = read_c_define(realm_defines_h, "REALM_PYTHON_LIB")
    realm_pyhome = os.path.dirname(os.path.dirname(realm_pylib.strip()[1:-1]))
    if curr_pyhome != realm_pyhome:
        print(
            "WARNING: Legate was compiled against the Python installation at "
            f"{realm_pyhome}, but you are currently using the Python "
            f"installation at {curr_pyhome}"
        )
    # If using NCCL prefer parallel launch mode over cooperative groups, as the
    # former plays better with Realm.
    cmd_env["NCCL_LAUNCH_MODE"] = "PARALLEL"
    # Make sure GASNet initializes MPI with the right level of
    # threading support
    cmd_env["GASNET_MPI_THREAD"] = "MPI_THREAD_MULTIPLE"
    # Set some environment variables depending on our configuration that we
    # will check in the Legate binary to ensure that it is properly configured
    # Always make sure we include the Legion library
    cmd_env[LIB_PATH] = os.pathsep.join(
        [legion_lib_path, legate_lib_path]
        + ([cmd_env[LIB_PATH]] if (LIB_PATH in cmd_env) else [])
    )

    if gpus > 0:
        assert "LEGATE_NEED_CUDA" not in cmd_env
        cmd_env["LEGATE_NEED_CUDA"] = str(1)

    if openmp > 0:
        assert "LEGATE_NEED_OPENMP" not in cmd_env
        cmd_env["LEGATE_NEED_OPENMP"] = str(1)

    if ranks > 1:
        assert "LEGATE_NEED_GASNET" not in cmd_env
        cmd_env["LEGATE_NEED_GASNET"] = str(1)

    if progress:
        assert "LEGATE_SHOW_PROGRESS" not in cmd_env
        cmd_env["LEGATE_SHOW_PROGRESS"] = str(1)
    if mem_usage:
        assert "LEGATE_SHOW_USAGE" not in cmd_env
        cmd_env["LEGATE_SHOW_USAGE"] = str(1)

    # Configure certain limits
    cmd_env["LEGATE_MAX_DIM"] = os.environ.get(
        "LEGATE_MAX_DIM"
    ) or read_c_define(legion_defines_h, "LEGION_MAX_DIM")
    cmd_env["LEGATE_MAX_FIELDS"] = os.environ.get(
        "LEGATE_MAX_FIELDS"
    ) or read_c_define(legion_defines_h, "LEGION_MAX_FIELDS")
    assert cmd_env["LEGATE_MAX_DIM"] is not None
    assert cmd_env["LEGATE_MAX_FIELDS"] is not None

    # Special run modes
    if freeze_on_error:
        cmd_env["LEGION_FREEZE_ON_ERROR"] = str(1)

    # Debugging options
    cmd_env["REALM_BACKTRACE"] = str(1)
    if gasnet_trace:
        cmd_env["GASNET_TRACEFILE"] = os.path.join(log_dir, "gasnet_%.log")

    # Add launcher
    if launcher == "mpirun":
        # TODO: $OMPI_COMM_WORLD_RANK will only work for OpenMPI and IBM
        # Spectrum MPI. Intel MPI and MPICH use $PMI_RANK, MVAPICH2 uses
        # $MV2_COMM_WORLD_RANK. Figure out which one to use based on the
        # output of `mpirun --version`.
        rank_id = "%q{OMPI_COMM_WORLD_RANK}"
        cmd = [
            "mpirun",
            "-n",
            str(ranks),
            "--npernode",
            str(ranks_per_node),
            "--bind-to",
            "none",
            "--mca",
            "mpi_warn_on_fork",
            "0",
        ]
        for var in cmd_env:
            if (
                var.endswith("PATH")
                or var.startswith("CONDA_")
                or var.startswith("LEGATE_")
                or var.startswith("LEGION_")
                or var.startswith("LG_")
                or var.startswith("REALM_")
                or var.startswith("GASNET_")
                or var.startswith("PYTHON")
                or var.startswith("UCX_")
                or var.startswith("NCCL_")
                or var.startswith("CUNUMERIC_")
                or var.startswith("NVIDIA_")
            ):
                cmd += ["-x", var]
    elif launcher == "jsrun":
        rank_id = "%q{OMPI_COMM_WORLD_RANK}"
        cmd = [
            "jsrun",
            "-n",
            str(ranks // ranks_per_node),
            "-r",
            "1",
            "-a",
            str(ranks_per_node),
            "-c",
            "ALL_CPUS",
            "-g",
            "ALL_GPUS",
            "-b",
            "none",
        ]
    elif launcher == "srun":
        rank_id = "%q{SLURM_PROCID}"
        cmd = [
            "srun",
            "-n",
            str(ranks),
            "--ntasks-per-node",
            str(ranks_per_node),
        ]
        if gdb or cuda_gdb:
            # Execute in pseudo-terminal mode when we need to be interactive
            cmd += ["--pty"]
    elif launcher == "none":
        rank_id = None
        if ranks == 1:
            rank_id = "0"
        else:
            for v in [
                "OMPI_COMM_WORLD_RANK",
                "PMI_RANK",
                "MV2_COMM_WORLD_RANK",
                "SLURM_PROCID",
            ]:
                if v in os.environ:
                    rank_id = os.environ[v]
                    break
        if rank_id is None:
            raise Exception(
                "Could not detect rank ID on multi-rank run with "
                "externally-managed launching (no --launcher provided). "
                "If you want Legate to use a launcher (e.g. mpirun) "
                "internally (recommended), then you need to tell us which one "
                "to use through --launcher. Otherwise you need to invoke the "
                "legate script itself through a launcher."
            )
        cmd = []
    else:
        raise Exception("Unsupported launcher: %s" % launcher)
    cmd += launcher_extra
    # Add any wrappers before the executable
    if any(f is not None for f in [cpu_bind, mem_bind, gpu_bind, nic_bind]):
        cmd += [
            bind_sh_path,
            "local" if launcher == "none" and ranks == 1 else launcher,
        ]
        if cpu_bind is not None:
            if len(cpu_bind.split("/")) != ranks_per_node:
                raise Exception(
                    "Number of groups in --cpu-bind not equal to "
                    "--ranks-per-node"
                )
            cmd += ["--cpus", cpu_bind]
        if gpu_bind is not None:
            if len(gpu_bind.split("/")) != ranks_per_node:
                raise Exception(
                    "Number of groups in --gpu-bind not equal to "
                    "--ranks-per-node"
                )
            cmd += ["--gpus", gpu_bind]
        if mem_bind is not None:
            if len(mem_bind.split("/")) != ranks_per_node:
                raise Exception(
                    "Number of groups in --mem-bind not equal to "
                    "--ranks-per-node"
                )
            cmd += ["--mems", mem_bind]
        if nic_bind is not None:
            if len(nic_bind.split("/")) != ranks_per_node:
                raise Exception(
                    "Number of groups in --nic-bind not equal to "
                    "--ranks-per-node"
                )
            cmd += ["--nics", nic_bind]
    if gdb:
        if ranks > 1:
            print("WARNING: Legate does not support gdb for multi-rank runs")
        elif os_name == "Darwin":
            cmd += ["lldb", "--"]
        else:
            cmd += ["gdb", "--args"]
    if cuda_gdb:
        if ranks > 1:
            print(
                "WARNING: Legate does not support cuda-gdb for multi-rank runs"
            )
        else:
            cmd += ["cuda-gdb", "--args"]
    if nvprof:
        cmd += [
            "nvprof",
            "-o",
            os.path.join(log_dir, "legate_%s.nvvp" % rank_id),
        ]
    if nsys:
        cmd += [
            "nsys",
            "profile",
            "-t",
            nsys_targets,
            "-o",
            os.path.join(log_dir, "legate_%s" % rank_id),
        ] + nsys_extra
        if "-s" not in nsys_extra:
            cmd += ["-s", "none"]
    # Add memcheck right before the binary
    if memcheck:
        cmd += ["cuda-memcheck"]
    # Now we're ready to build the actual command to run
    cmd += [legion_python]
    # This has to go before script name
    if not_control_replicable:
        cmd += ["--nocr"]
    if module is not None:
        cmd += ["-m", str(module)]
    # We always need one python processor per rank and no local fields
    cmd += ["-ll:py", "1", "-lg:local", "0"]
    # Special run modes
    if freeze_on_error or gdb or cuda_gdb:
        # Running with userspace threads would not allow us to inspect the
        # stacktraces of suspended threads.
        cmd += ["-ll:force_kthreads"]
    # Translate the requests to Realm command line parameters
    if cpus != 1:
        cmd += ["-ll:cpu", str(cpus)]
    if gpus > 0:
        # Make sure that we skip busy GPUs
        cmd += ["-ll:gpu", str(gpus), "-cuda:skipbusy"]
    if openmp > 0:
        if ompthreads > 0:
            cmd += ["-ll:ocpu", str(openmp), "-ll:othr", str(ompthreads)]
            # If we have support for numa memories then add the extra flag
            if numamem > 0:
                cmd += ["-ll:onuma", "1"]
            else:
                cmd += ["-ll:onuma", "0"]
        else:
            print(
                "WARNING: Legate is ignoring request for "
                + str(openmp)
                + "OpenMP processors with 0 threads"
            )
    if utility != 1:
        cmd += ["-ll:util", str(utility)]
        # If we are running multi-rank then make the number of active
        # message handler threads equal to our number of utility
        # processors in order to prevent head-of-line blocking
        if ranks > 1:
            cmd += ["-ll:bgwork", str(utility)]
    # Always specify the csize
    cmd += ["-ll:csize", str(sysmem)]
    if numamem > 0:
        cmd += ["-ll:nsize", str(numamem)]
    # Only specify GPU memory sizes if we have GPUs
    if gpus > 0:
        cmd += ["-ll:fsize", str(fbmem), "-ll:zsize", str(zcmem)]
    if regmem > 0:
        cmd += ["-ll:rsize", str(regmem)]
    logging_levels = ["openmp=5"]
    if profile:
        cmd += [
            "-lg:prof",
            str(ranks),
            "-lg:prof_logfile",
            os.path.join(log_dir, "legate_%.prof"),
        ]
        logging_levels.append("legion_prof=2")
    # The gpu log supression may not be needed in the future.
    # Currently, the cuda hijack layer generates some spurious warnings.
    if gpus > 0:
        logging_levels.append("gpu=5")
    if dataflow or event:
        cmd += ["-lg:spy"]
        logging_levels.append("legion_spy=2")
        # Spy output is dumped to the same place as other logging, so we must
        # redirect all logging to a file, even if the user didn't ask for it.
        if user_logging_levels is not None and not log_to_file:
            print(
                "WARNING: Logging output is being redirected to a file in "
                f"directory {log_dir}"
            )
        log_to_file = True
    logging_levels = ",".join(logging_levels)
    if user_logging_levels is not None:
        logging_levels += "," + user_logging_levels
    cmd += ["-level", logging_levels]
    if log_to_file:
        cmd += ["-logfile", os.path.join(log_dir, "legate_%.log")]
    if gdb and os_name == "Darwin":
        print(
            "WARNING: You must start the debugging session with the following "
            "command as LLDB "
        )
        print(
            "no longer forwards the environment to subprocesses for security "
            "reasons:"
        )
        print()
        print(
            "(lldb) process launch -v "
            + LIB_PATH
            + "="
            + cmd_env[LIB_PATH]
            + " -v PYTHONPATH="
            + cmd_env["PYTHONPATH"]
        )
        print()

    cmd += ["-lg:eager_alloc_percentage", eager_alloc]

    # Append all user flags to the command so that they can override whatever
    # the launcher has come up with.
    if opts:
        cmd += opts

    # Create output directory
    os.makedirs(log_dir, exist_ok=True)
    # Launch the child process
    if verbose and (launcher != "none" or rank_id == "0"):
        print(
            "Running: " + " ".join([shlex.quote(t) for t in cmd]), flush=True
        )
    child_proc = subprocess.Popen(cmd, env=cmd_env)
    # Wait for it to finish running
    result = child_proc.wait()
    # If we're profiling post process the logfiles and then clean them up when
    # we're done; make sure we only do this once if on a multi-rank run with
    # externally-managed launching
    if profile and (launcher != "none" or rank_id == "0"):
        prof_cmd = [str(legion_prof_py), "-o", "legate_prof"]
        for n in range(ranks):
            prof_cmd += ["legate_" + str(n) + ".prof"]
        if ranks // ranks_per_node > 4:
            print(
                "Skipping the processing of profiler output, to avoid wasting "
                "resources in a large allocation. Please manually run: "
                + " ".join([shlex.quote(t) for t in prof_cmd]),
                flush=True,
            )
            keep_logs = True
        else:
            if verbose:
                print(
                    "Running: " + " ".join([shlex.quote(t) for t in prof_cmd]),
                    flush=True,
                )
            subprocess.check_call(prof_cmd, cwd=log_dir)
        if not keep_logs:
            # Clean up our mess of Legion Prof files
            for n in range(ranks):
                os.remove(os.path.join(log_dir, "legate_" + str(n) + ".prof"))
    # Similarly for spy runs
    if (dataflow or event) and (launcher != "none" or rank_id == "0"):
        spy_cmd = [str(legion_spy_py)]
        if dataflow and event:
            spy_cmd += ["-de"]
        elif dataflow:
            spy_cmd += ["-d"]
        else:
            spy_cmd += ["-e"]
        for n in range(ranks):
            spy_cmd += ["legate_" + str(n) + ".log"]
        if ranks // ranks_per_node > 4:
            print(
                "Skipping the processing of spy output, to avoid wasting "
                "resources in a large allocation. Please manually run: "
                + " ".join([shlex.quote(t) for t in spy_cmd]),
                flush=True,
            )
            keep_logs = True
        else:
            if verbose:
                print(
                    "Running: " + " ".join([shlex.quote(t) for t in spy_cmd]),
                    flush=True,
                )
            subprocess.check_call(spy_cmd, cwd=log_dir)
        if user_logging_levels is None and not keep_logs:
            # Clean up our mess of Legion Spy files, unless the user is doing
            # some extra logging, in which case theirs and Spy's logs will be
            # in the same file.
            for n in range(ranks):
                os.remove(os.path.join(log_dir, "legate_" + str(n) + ".log"))
    return result


def driver():
    parser = argparse.ArgumentParser(
        description="Legate Driver.", allow_abbrev=False
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        dest="nodes",
        help="Number of nodes to use",
    )
    parser.add_argument(
        "--ranks-per-node",
        type=int,
        default=1,
        dest="ranks_per_node",
        help="Number of ranks (processes running copies of the program) to "
        "launch per node. The default (1 rank per node) will typically result "
        "in the best performance.",
    )
    parser.add_argument(
        "--no-replicate",
        dest="not_control_replicable",
        action="store_true",
        required=False,
        help="Execute this program without control replication.  Most of the "
        "time, this is not recommended.  This option should be used for "
        "debugging.  The -lg:safe_ctrlrepl Legion option may be helpful "
        "with discovering issues with replicated control.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        dest="cpus",
        help="Number of CPUs to use per rank",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        dest="gpus",
        help="Number of GPUs to use per rank",
    )
    parser.add_argument(
        "--omps",
        type=int,
        default=(int(os.environ.get("LEGATE_OMP_PROCS", 0))),
        dest="openmp",
        help="Number of OpenMP groups to use per rank",
    )
    parser.add_argument(
        "--ompthreads",
        type=int,
        default=(int(os.environ.get("LEGATE_OMP_THREADS", 4))),
        dest="ompthreads",
        help="Number of threads per OpenMP group",
    )
    parser.add_argument(
        "--utility",
        type=int,
        default=(int(os.environ.get("LEGATE_UTILITY_CORES", 2))),
        dest="utility",
        help="Number of Utility processors per rank to request for meta-work",
    )
    parser.add_argument(
        "--sysmem",
        type=int,
        default=(int(os.environ.get("LEGATE_SYSMEM", 4000))),
        dest="sysmem",
        help="Amount of DRAM memory per rank (in MBs)",
    )
    parser.add_argument(
        "--numamem",
        type=int,
        default=(int(os.environ.get("LEGATE_NUMAMEM", 0))),
        dest="numamem",
        help="Amount of DRAM memory per NUMA domain per rank (in MBs)",
    )
    parser.add_argument(
        "--fbmem",
        type=int,
        default=(int(os.environ.get("LEGATE_FBMEM", 4000))),
        dest="fbmem",
        help="Amount of framebuffer memory per GPU (in MBs)",
    )
    parser.add_argument(
        "--zcmem",
        type=int,
        default=(int(os.environ.get("LEGATE_ZCMEM", 32))),
        dest="zcmem",
        help="Amount of zero-copy memory per rank (in MBs)",
    )
    parser.add_argument(
        "--regmem",
        type=int,
        default=(int(os.environ.get("LEGATE_REGMEM", 0))),
        dest="regmem",
        help="Amount of registered CPU-side pinned memory per rank (in MBs)",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        required=False,
        help="profile Legate execution",
    )
    parser.add_argument(
        "--freeze-on-error",
        dest="freeze_on_error",
        action="store_true",
        required=False,
        help="if the program crashes, freeze execution right before exit so a "
        "debugger can be attached",
    )
    parser.add_argument(
        "--dataflow",
        dest="dataflow",
        action="store_true",
        required=False,
        help="Generate Legate dataflow graph",
    )
    parser.add_argument(
        "--event",
        dest="event",
        action="store_true",
        required=False,
        help="Generate Legate event graph",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.getcwd(),
        dest="logdir",
        help="Directory for Legate log files (automatically created if it "
        "doesn't exist; defaults to current directory)",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default=None,
        dest="user_logging_levels",
        help="extra loggers to turn on",
    )
    parser.add_argument(
        "--log-to-file",
        dest="log_to_file",
        action="store_true",
        required=False,
        help="redirect logging output to a file inside --logdir",
    )
    parser.add_argument(
        "--keep-logs",
        dest="keep_logs",
        action="store_true",
        required=False,
        help="don't delete profiler & spy dumps after processing",
    )
    parser.add_argument(
        "--gdb",
        dest="gdb",
        action="store_true",
        required=False,
        help="run Legate inside gdb",
    )
    parser.add_argument(
        "--cuda-gdb",
        dest="cuda_gdb",
        action="store_true",
        required=False,
        help="run Legate inside cuda-gdb",
    )
    parser.add_argument(
        "--memcheck",
        dest="memcheck",
        action="store_true",
        required=False,
        help="run Legate with cuda-memcheck",
    )
    parser.add_argument(
        "--module",
        dest="module",
        default=None,
        required=False,
        help="Specify a Python module to load before running",
    )
    parser.add_argument(
        "--nvprof",
        dest="nvprof",
        action="store_true",
        required=False,
        help="run Legate with nvprof",
    )
    parser.add_argument(
        "--nsys",
        dest="nsys",
        action="store_true",
        required=False,
        help="run Legate with Nsight Systems",
    )
    parser.add_argument(
        "--nsys-targets",
        dest="nsys_targets",
        default="cublas,cuda,cudnn,nvtx,ucx",
        required=False,
        help="Specify profiling targets for Nsight Systems",
    )
    parser.add_argument(
        "--nsys-extra",
        dest="nsys_extra",
        action="append",
        default=[],
        required=False,
        help="Specify extra flags for Nsight Systems",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        required=False,
        help="show progress of operations when running the program",
    )
    parser.add_argument(
        "--mem-usage",
        dest="mem_usage",
        action="store_true",
        required=False,
        help="report the memory usage by Legate in every memory",
    )
    parser.add_argument(
        "--launcher",
        dest="launcher",
        choices=["mpirun", "jsrun", "srun", "none"],
        default="none",
        help='launcher program to use (set to "none" for local runs, or if '
        "the launch has already happened by the time legate is invoked)",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="print out each shell command before running it",
    )
    parser.add_argument(
        "--gasnet-trace",
        dest="gasnet_trace",
        action="store_true",
        default=False,
        required=False,
        help="enable GASNet tracing (assumes GASNet was configured with "
        "--enable--trace)",
    )
    # FIXME: We set the eager pool size to 50% of the total size for now.
    #        This flag will be gone once we roll out a new allocation scheme.
    parser.add_argument(
        "--eager-alloc-percentage",
        dest="eager_alloc",
        default=(os.environ.get("LEGATE_EAGER_ALLOC_PERCENTAGE", "50")),
        required=False,
        help="Specify the size of eager allocation pool in percentage",
    )
    parser.add_argument(
        "--cpu-bind",
        help="CPU cores to bind each rank to. Comma-separated core IDs as "
        "well as ranges are accepted, as reported by `numactl`. Binding "
        "instructions for all ranks should be listed in one string, separated "
        "by `/`.",
    )
    parser.add_argument(
        "--mem-bind",
        help="NUMA memories to bind each rank to. Use comma-separated integer "
        "IDs as reported by `numactl`. Binding instructions for all ranks "
        "should be listed in one string, separated by `/`.",
    )
    parser.add_argument(
        "--gpu-bind",
        help="GPUs to bind each rank to. Use comma-separated integer IDs as "
        "reported by `nvidia-smi`. Binding instructions for all ranks "
        "should be listed in one string, separated by `/`.",
    )
    parser.add_argument(
        "--nic-bind",
        help="NICs to bind each rank to. Use comma-separated device names as "
        "appropriate for the GASNet conduit in use. Binding instructions for "
        "all ranks should be listed in one string, separated by `/`.",
    )
    parser.add_argument(
        "--launcher-extra",
        dest="launcher_extra",
        action="append",
        default=[],
        required=False,
        help="additional argument to pass to the launcher (can appear more "
        "than once)",
    )
    args, opts = parser.parse_known_args()
    # See if we have at least one script file to run
    console = True
    for opt in opts:
        if ".py" in opt:
            console = False
            break
    if console and not args.not_control_replicable:
        print("WARNING: Disabling control replication for interactive run")
        args.not_control_replicable = True
    ranks = args.nodes * args.ranks_per_node
    return run_legate(
        ranks,
        args.ranks_per_node,
        args.cpus,
        args.gpus,
        args.openmp,
        args.ompthreads,
        args.utility,
        args.sysmem,
        args.numamem,
        args.fbmem,
        args.zcmem,
        args.regmem,
        opts,
        args.profile,
        args.dataflow,
        args.event,
        args.logdir,
        args.user_logging_levels,
        args.log_to_file,
        args.keep_logs,
        args.gdb,
        args.cuda_gdb,
        args.memcheck,
        args.module,
        args.nvprof,
        args.nsys,
        args.nsys_targets,
        args.nsys_extra,
        args.progress,
        args.freeze_on_error,
        args.mem_usage,
        args.not_control_replicable,
        args.launcher,
        args.verbose,
        args.gasnet_trace,
        args.eager_alloc,
        args.cpu_bind,
        args.mem_bind,
        args.gpu_bind,
        args.nic_bind,
        args.launcher_extra,
    )


if __name__ == "__main__":
    sys.exit(driver())
