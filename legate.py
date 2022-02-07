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

from __future__ import print_function

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys

_version = sys.version_info.major

try:
    _input = raw_input  # Python 2.x:
except NameError:
    _input = input  # Python 3.x:

os_name = platform.system()

if os_name == "Linux":
    dylib_ext = ".so"
    LIB_PATH = "LD_LIBRARY_PATH"
elif os_name == "Darwin":  # Don't currently support Darwin at the moment
    dylib_ext = ".dylib"
    LIB_PATH = "DYLD_LIBRARY_PATH"
else:
    raise Exception("Legate does not work on %s" % platform.system())


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


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


def read_conduit(legate_dir):
    realm_defines = os.path.join(legate_dir, "include", "realm_defines.h")
    for conduit in ["ibv", "ucx", "aries", "mpi", "udp"]:
        if read_c_define(realm_defines, f"GASNET_CONDUIT_{conduit.upper()}"):
            return conduit
    raise Exception("Could not detect a supported GASNet conduit")


def find_python_module(legate_dir):
    lib_dir = os.path.join(legate_dir, "lib")
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
    gdb,
    cuda_gdb,
    memcheck,
    module,
    nvprof,
    nsys,
    progress,
    freeze_on_error,
    no_tensor_cores,
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
    # Build the environment for the subprocess invocation
    legate_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    cmd_env = dict(os.environ.items())
    env_json = load_json_config(
        os.path.join(legate_dir, "share", ".legate-env.json")
    )
    if env_json is not None:
        append_vars = env_json.get("APPEND_VARS", [])
        append_vars_tuplified = [tuple(var) for var in append_vars]
        for (k, v) in append_vars_tuplified:
            if k not in cmd_env:
                cmd_env[k] = v
            else:
                cmd_env[k] = cmd_env[k] + os.pathsep + v
        vars = env_json.get("VARS", [])
        vars_tuplified = [tuple(var) for var in vars]
        for (k, v) in vars_tuplified:
            cmd_env[k] = v
    libs_json = load_json_config(
        os.path.join(legate_dir, "share", ".legate-libs.json")
    )
    if libs_json is not None:
        for lib_dir in libs_json.values():
            if LIB_PATH not in cmd_env:
                cmd_env[LIB_PATH] = os.path.join(lib_dir, "lib")
            else:
                cmd_env[LIB_PATH] = (
                    os.path.join(lib_dir, "lib")
                    + os.pathsep
                    + cmd_env[LIB_PATH]
                )
    # We never want to save python byte code for legate
    cmd_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Set the path to the Legate module as an environment variable
    # The current directory should be added to PYTHONPATH as well
    if "PYTHONPATH" in cmd_env:
        cmd_env["PYTHONPATH"] += os.pathsep + ""
    else:
        cmd_env["PYTHONPATH"] = ""
    cmd_env["PYTHONPATH"] += os.pathsep + find_python_module(legate_dir)
    # Make sure the version of Python used by Realm is the same as what the
    # user is using currently.
    curr_pyhome = os.path.dirname(os.path.dirname(sys.executable))
    realm_defines = os.path.join(legate_dir, "include", "realm_defines.h")
    realm_pylib = read_c_define(realm_defines, "REALM_PYTHON_LIB")
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
    # Set some environment variables depending on our configuration that we
    # will check in the Legate binary to ensure that it is properly configured
    # Always make sure we include the Legion library
    if LIB_PATH not in cmd_env:
        cmd_env[LIB_PATH] = os.path.join(legate_dir, "lib")
    else:
        cmd_env[LIB_PATH] = (
            os.path.join(legate_dir, "lib") + os.pathsep + cmd_env[LIB_PATH]
        )
    cuda_config = os.path.join(legate_dir, "share", "legate", ".cuda.json")
    cuda_dir = load_json_config(cuda_config)
    if gpus > 0 and cuda_dir is None:
        raise ValueError(
            "Requested execution with GPUs but "
            + "Legate was not built with GPU support"
        )
    if gpus > 0:
        assert "LEGATE_NEED_CUDA" not in cmd_env
        cmd_env["LEGATE_NEED_CUDA"] = str(1)
        cmd_env[LIB_PATH] += os.pathsep + os.path.join(cuda_dir, "lib")
        cmd_env[LIB_PATH] += os.pathsep + os.path.join(cuda_dir, "lib64")
    if openmp > 0:
        assert "LEGATE_NEED_OPENMP" not in cmd_env
        cmd_env["LEGATE_NEED_OPENMP"] = str(1)
    if ranks > 1:
        assert "LEGATE_NEED_GASNET" not in cmd_env
        cmd_env["LEGATE_NEED_GASNET"] = str(1)
    if progress:
        assert "LEGATE_SHOW_PROGREES" not in cmd_env
        cmd_env["LEGATE_SHOW_PROGRESS"] = str(1)
    if no_tensor_cores:
        assert "LEGATE_DISABLE_TENSOR_CORES" not in cmd_env
        cmd_env["LEGATE_DISABLE_TENSOR_CORES"] = str(1)
    if mem_usage:
        assert "LEGATE_SHOW_USAGE" not in cmd_env
        cmd_env["LEGATE_SHOW_USAGE"] = str(1)
    # Configure certain limits
    defines_path = os.path.join(
        os.path.join(legate_dir, "include"), "legion_defines.h"
    )
    if "LEGATE_MAX_DIM" not in os.environ:
        cmd_env["LEGATE_MAX_DIM"] = read_c_define(
            defines_path, "LEGION_MAX_DIM"
        )
        assert cmd_env["LEGATE_MAX_DIM"] is not None
    if "LEGATE_MAX_FIELDS" not in os.environ:
        cmd_env["LEGATE_MAX_FIELDS"] = read_c_define(
            defines_path, "LEGION_MAX_FIELDS"
        )
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
                var == LIB_PATH
                or var.startswith("LEGATE_")
                or var.startswith("LEGION_")
                or var.startswith("LG_")
                or var.startswith("REALM_")
                or var.startswith("GASNET_")
                or var.startswith("PYTHON")
                or var.startswith("UCX_")
                or var.startswith("NCCL_")
                or var.startswith("CUNUMERIC_")
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
                "externally-managed launching"
            )
        cmd = []
    else:
        raise Exception("Unsupported launcher: %s" % launcher)
    cmd += launcher_extra
    # Add any wrappers before the executable
    binary_dir = os.path.join(legate_dir, "bin")
    if any(f is not None for f in [cpu_bind, mem_bind, gpu_bind, nic_bind]):
        conduit = read_conduit(legate_dir)
        cmd += [os.path.join(binary_dir, "bind.sh"), launcher, conduit]
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
            "cublas,cuda,cudnn,nvtx",
            "-s",
            "none",
            "-o",
            os.path.join(log_dir, "legate_%s" % rank_id),
        ]
    # Add memcheck right before the binary
    if memcheck:
        cmd += ["cuda-memcheck"]
    # Now we're ready to build the actual command to run
    cmd += [os.path.join(binary_dir, "legion_python")]
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
        cmd += [
            "-lg:spy",
            "-logfile",
            os.path.join(log_dir, "legate_%.spy"),
        ]
        logging_levels.append("legion_spy=2")
    logging_levels = ",".join(logging_levels)
    if user_logging_levels is not None:
        logging_levels += "," + user_logging_levels
    cmd += ["-level", logging_levels]
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
        tools_dir = os.path.join(legate_dir, "share", "legate")
        prof_py = os.path.join(tools_dir, "legion_prof.py")
        prof_cmd = [str(prof_py), "-o", "legate_prof"]
        for n in range(ranks):
            prof_cmd += ["legate_" + str(n) + ".prof"]
        if ranks // ranks_per_node > 4:
            print(
                "Skipping the processing of profiler output, to avoid wasting "
                "resources in a large allocation. Please manually run: "
                + " ".join([shlex.quote(t) for t in prof_cmd]),
                flush=True,
            )
        else:
            if verbose:
                print(
                    "Running: " + " ".join([shlex.quote(t) for t in prof_cmd]),
                    flush=True,
                )
            subprocess.check_call(prof_cmd, cwd=log_dir)
            # Clean up our mess of Legion Prof files
            for n in range(ranks):
                os.remove(os.path.join(log_dir, "legate_" + str(n) + ".prof"))
    # Similarly for spy runs
    if (dataflow or event) and (launcher != "none" or rank_id == "0"):
        tools_dir = os.path.join(legate_dir, "share", "legate")
        spy_py = os.path.join(tools_dir, "legion_spy.py")
        spy_cmd = [str(spy_py)]
        if dataflow and event:
            spy_cmd += ["-de"]
        elif dataflow:
            spy_cmd += ["-d"]
        else:
            spy_cmd += ["-e"]
        for n in range(ranks):
            spy_cmd += ["legate_" + str(n) + ".spy"]
        if ranks // ranks_per_node > 4:
            print(
                "Skipping the processing of spy output, to avoid wasting "
                "resources in a large allocation. Please manually run: "
                + " ".join([shlex.quote(t) for t in spy_cmd]),
                flush=True,
            )
        else:
            if verbose:
                print(
                    "Running: " + " ".join([shlex.quote(t) for t in spy_cmd]),
                    flush=True,
                )
            subprocess.check_call(spy_cmd, cwd=log_dir)
            # Clean up our mess of Legion Spy files
            for n in range(ranks):
                os.remove(os.path.join(log_dir, "legate_" + str(n) + ".spy"))
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
        help="Directory for Legate log files (defaults to current directory)",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default=None,
        dest="user_logging_levels",
        help="extra loggers to turn on",
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
        help="run Legate with nsys",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        required=False,
        help="show progress of operations when running the program",
    )
    parser.add_argument(
        "--no-tensor",
        dest="no_tensor_cores",
        action="store_true",
        required=False,
        help="disable the use of GPU tensor cores for better determinism",
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
        args.gdb,
        args.cuda_gdb,
        args.memcheck,
        args.module,
        args.nvprof,
        args.nsys,
        args.progress,
        args.freeze_on_error,
        args.no_tensor_cores,
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
