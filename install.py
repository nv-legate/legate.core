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
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
from distutils import sysconfig

import setuptools

# Flush output on newlines
sys.stdout.reconfigure(line_buffering=True)

os_name = platform.system()

required_thrust_version = "cuda-11.2"

# Work around breaking change in setuptools 60
setup_py_flags = []
if int(setuptools.__version__.split(".")[0]) >= 60:
    setup_py_flags = ["--single-version-externally-managed", "--root=/"]


class BooleanFlag(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default,
        required=False,
        help="",
        metavar=None,
    ):
        assert all(not opt.startswith("--no") for opt in option_strings)

        def flatten(list):
            return [item for sublist in list for item in sublist]

        option_strings = flatten(
            [
                [opt, "--no-" + opt[2:], "--no" + opt[2:]]
                if opt.startswith("--")
                else [opt]
                for opt in option_strings
            ]
        )
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            const=None,
            default=default,
            type=bool,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, not option_string.startswith("--no"))


def execute_command(args, verbose, **kwargs):
    if verbose:
        print('Executing: "', " ".join(args), '" with ', kwargs)
    subprocess.check_call(args, **kwargs)


def execute_command_check_output(args, verbose, **kwargs):
    if verbose:
        print('Executing: "', " ".join(args), '" with ', kwargs)
    return subprocess.check_output(args, **kwargs)


def find_active_python_version_and_path():
    # Launching a sub-process to do this in a general way seems hard
    version = (
        str(sys.version_info.major)
        + "."
        + str(sys.version_info.minor)
        + "."
        + str(sys.version_info.micro)
    )
    cv = sysconfig.get_config_vars()
    paths = [os.path.join(cv[p], cv["LDLIBRARY"]) for p in ("LIBDIR", "LIBPL")]
    # ensure that static libraries are replaced with the dynamic version
    paths = [
        os.path.splitext(p)[0] + (".dylib" if os_name == "Darwin" else ".so")
        for p in paths
    ]
    paths = [p for p in paths if os.path.isfile(p)]
    e = "Error: could not auto-locate python library."
    assert paths, e
    return version, paths[0]


def install_legion_python(legion_src_dir, install_dir, verbose=False):

    os.makedirs(os.path.join(install_dir, "share", "legate"), exist_ok=True)

    src = os.path.join(legion_src_dir, "runtime", "legion", "legion_c_util.h")
    dst = os.path.join(install_dir, "include", "legion", "legion_c_util.h")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(
        src
    ):
        execute_command(["cp", src, dst], verbose)

    execute_command(
        [
            "cp",
            "legion_spy.py",
            os.path.join(install_dir, "share", "legate", "legion_spy.py"),
        ],
        verbose,
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    execute_command(
        [
            "cp",
            "legion_prof.py",
            os.path.join(install_dir, "share", "legate", "legion_prof.py"),
        ],
        verbose,
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    execute_command(
        [
            "cp",
            "legion_serializer.py",
            os.path.join(
                install_dir, "share", "legate", "legion_serializer.py"
            ),
        ],
        verbose,
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    execute_command(
        [
            "cp",
            "legion_prof_copy.html.template",
            os.path.join(
                install_dir,
                "share",
                "legate",
                "legion_prof_copy.html.template",
            ),
        ],
        verbose,
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    execute_command(
        [
            "cp",
            "-r",
            "legion_prof_files",
            os.path.join(install_dir, "share", "legate", "legion_prof_files"),
        ],
        verbose,
        cwd=os.path.join(legion_src_dir, "tools"),
    )


def get_legion_src_dir(legion_dir, build_dir, verbose=False):
    if not legion_dir:
        return os.path.join(build_dir, "_deps", "legion-src")
    elif os.path.exists(os.path.join(legion_dir, "CMakeCache.txt")):
        # Otherwise, assume `legion_dir` is the path to a CMake build dir
        src_dir = (
            execute_command_check_output(
                [
                    "grep",
                    "--color=never",
                    "Legion_SOURCE_DIR",
                    os.path.join(legion_dir, "CMakeCache.txt"),
                ],
                verbose,
            )
            .decode("UTF-8")
            .strip()
        )
        # `src_dir` will be something like:
        # `Legion_SOURCE_DIR:STATIC=/path/to/legion`
        src_dir = src_dir.split("=")[1]
        return src_dir
    else:
        raise Exception("Could not determine path to Legion source directory")


def install(
    gasnet,
    cuda,
    arch,
    openmp,
    march,
    hdf,
    llvm,
    spy,
    conduit,
    nccl_dir,
    cmake_exe,
    cmake_generator,
    install_dir,
    gasnet_dir,
    pylib_name,
    cuda_dir,
    maxdim,
    maxfields,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    extra_flags,
    thread_count,
    verbose,
    thrust_dir,
    legion_dir,
    legion_url,
    legion_branch,
    unknown,
):
    legate_core_dir = os.path.dirname(os.path.realpath(__file__))

    if pylib_name is None:
        pyversion, pylib_name = find_active_python_version_and_path()
    else:
        f_name = os.path.split(pylib_name)[-1]
        match = re.match(r"^libpython(\d\d?\.\d\d?)", f_name)
        e = "Unable to get version from library name {}".format(pylib_name)
        assert match, e
        pyversion = match.group(1)
    print("Using python lib and version: {}, {}".format(pylib_name, pyversion))

    if install_dir is None:
        install_dir = os.path.join(legate_core_dir, "install")
    install_dir = os.path.realpath(install_dir)

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    if thrust_dir is not None:
        thrust_dir = os.path.realpath(thrust_dir)

    build_dir = os.path.join(legate_core_dir, "_skbuild")

    if clean_first:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(
            os.path.join(legate_core_dir, "dist"), ignore_errors=True
        )
        shutil.rmtree(
            os.path.join(legate_core_dir, "legate.core.egg-info"),
            ignore_errors=True,
        )

    if legion_dir is not None:
        if os.path.exists(os.path.join(legion_dir, "CMakeCache.txt")):
            # Install Legion if legion_dir is a path to its build dir
            execute_command(
                [cmake_exe, "--install", legion_dir, "--prefix", install_dir],
                verbose,
            )

    # Configure and build legate.core via setup.py
    setup_py_cmd = [
        sys.executable,
        "setup.py",
        "-j",
        str(thread_count),
    ]
    if cmake_generator:
        setup_py_cmd += ["-G", cmake_generator, "--skip-generator-test"]

    setup_py_cmd += ["install", "--force"] + setup_py_flags

    if unknown is not None:
        try:
            prefix_loc = unknown.index("--prefix")
            setup_py_cmd.extend(unknown[prefix_loc : prefix_loc + 2])
        except ValueError:
            setup_py_cmd += ["--prefix", str(install_dir)]
    else:
        setup_py_cmd += ["--prefix", str(install_dir)]

    cmake_flags = ["--"]

    if debug or verbose:
        cmake_flags += ["--log-level=%s" % ("DEBUG" if debug else "VERBOSE")]

    cmake_flags += f"""\
-DCMAKE_BUILD_TYPE={(
    "Debug" if debug else "RelWithDebInfo" if debug_release else "Release"
)}
-DBUILD_SHARED_LIBS=ON
-DBUILD_MARCH={march}
-DCMAKE_CUDA_ARCHITECTURES={arch}
-DLegion_MAX_DIM={str(maxdim)}
-DLegion_MAX_FIELDS={str(maxfields)}
-DLegion_SPY={("ON" if spy else "OFF")}
-DLegion_BOUNDS_CHECKS={("ON" if check_bounds else "OFF")}
-DLegion_USE_CUDA={("ON" if cuda else "OFF")}
-DLegion_USE_OpenMP={("ON" if openmp else "OFF")}
-DLegion_USE_LLVM={("ON" if llvm else "OFF")}
-DLegion_USE_GASNet={("ON" if gasnet else "OFF")}
-DLegion_USE_HDF5={("ON" if hdf else "OFF")}
-DLegion_USE_Python=ON
-DLegion_Python_Version={pyversion}
-DLegion_REDOP_COMPLEX=ON
-DLegion_REDOP_HALF=ON
-DLegion_BUILD_BINDINGS=ON
""".splitlines()

    if nccl_dir:
        cmake_flags += ["-DNCCL_DIR=%s" % nccl_dir]
    if gasnet_dir:
        cmake_flags += ["-DGASNet_ROOT_DIR=%s" % gasnet_dir]
    if conduit:
        cmake_flags += ["-DGASNet_CONDUIT=%s" % conduit]
    if cuda_dir:
        cmake_flags += ["-DCUDA_TOOLKIT_ROOT_DIR=%s" % cuda_dir]
    if thrust_dir:
        cmake_flags += ["-DThrust_ROOT=%s" % thrust_dir]
    if legion_dir:
        cmake_flags += ["-DLegion_ROOT=%s" % legion_dir]
    if legion_url:
        cmake_flags += ["-DLEGATE_CORE_LEGION_REPOSITORY=%s" % legion_url]
    if legion_branch:
        cmake_flags += ["-DLEGATE_CORE_LEGION_BRANCH=%s" % legion_branch]

    setup_py_cmd += cmake_flags

    setup_py_cmd += ["--"]
    setup_py_cmd += ["-j", str(thread_count)]

    # run python setup.py <args>
    execute_command(setup_py_cmd, verbose, cwd=legate_core_dir)

    legion_src_dir = get_legion_src_dir(legion_dir, build_dir, verbose)
    install_legion_python(legion_src_dir, install_dir, verbose)


def driver():
    parser = argparse.ArgumentParser(description="Install Legate front end.")
    parser.add_argument(
        "--install-dir",
        dest="install_dir",
        metavar="DIR",
        required=False,
        default=None,
        help="Path to install all Legate-related software",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG", "0") == "1",
        help="Build Legate and Legion with no optimizations, and full "
        "debugging checks.",
    )
    parser.add_argument(
        "--debug-release",
        dest="debug_release",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG_RELEASE", "0") == "1",
        help="Build Legate and Legion with optimizations enabled, but include "
        "debugging symbols.",
    )
    parser.add_argument(
        "--check-bounds",
        dest="check_bounds",
        action="store_true",
        required=False,
        default=os.environ.get("CHECK_BOUNDS", "0") == "1",
        help="Build Legion with bounds checking enabled (warning: expensive).",
    )
    parser.add_argument(
        "--max-dim",
        dest="maxdim",
        type=int,
        default=int(os.environ.get("LEGION_MAX_DIM", 4)),
        help="Maximum number of dimensions that Legate will support",
    )
    parser.add_argument(
        "--max-fields",
        dest="maxfields",
        type=int,
        default=int(os.environ.get("LEGION_MAX_FIELDS", 256)),
        help="Maximum number of fields that Legate will support",
    )
    parser.add_argument(
        "--gasnet",
        dest="gasnet",
        action="store_true",
        required=False,
        default=os.environ.get("USE_GASNET", "0") == "1",
        help="Build Legate with GASNet.",
    )
    parser.add_argument(
        "--with-gasnet",
        dest="gasnet_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("GASNET"),
        help="Path to GASNet installation directory.",
    )
    parser.add_argument(
        "--cuda",
        action=BooleanFlag,
        default=os.environ.get("USE_CUDA", "0") == "1",
        help="Build Legate with CUDA support.",
    )
    parser.add_argument(
        "--with-cuda",
        dest="cuda_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CUDA"),
        help="Path to CUDA installation directory.",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        action="store",
        required=False,
        default="NATIVE",
        help="Specify the target GPU architecture.",
    )
    parser.add_argument(
        "--openmp",
        action=BooleanFlag,
        default=os.environ.get("USE_OPENMP", "0") == "1",
        help="Build Legate with OpenMP support.",
    )
    parser.add_argument(
        "--march",
        dest="march",
        required=False,
        default="native",
        help="Specify the target CPU architecture.",
    )
    parser.add_argument(
        "--llvm",
        dest="llvm",
        action="store_true",
        required=False,
        default=os.environ.get("USE_LLVM", "0") == "1",
        help="Build Legate with LLVM support.",
    )
    parser.add_argument(
        "--hdf5",
        "--hdf",
        dest="hdf",
        action="store_true",
        required=False,
        default=os.environ.get("USE_HDF", "0") == "1",
        help="Build Legate with HDF support.",
    )
    parser.add_argument(
        "--spy",
        dest="spy",
        action="store_true",
        required=False,
        default=os.environ.get("USE_SPY", "0") == "1",
        help="Build Legate with detailed Legion Spy enabled.",
    )
    parser.add_argument(
        "--conduit",
        dest="conduit",
        action="store",
        required=False,
        choices=["ibv", "ucx", "aries", "mpi", "udp"],
        default=os.environ.get("CONDUIT"),
        help="Build Legate with specified GASNet conduit.",
    )
    parser.add_argument(
        "--with-nccl",
        dest="nccl_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("NCCL_PATH"),
        help="Path to NCCL installation directory.",
    )
    parser.add_argument(
        "--python-lib",
        dest="pylib_name",
        action="store",
        required=False,
        default=None,
        help=(
            "Build Legate against the specified Python shared library. "
            "Default is to use the Python library currently executing this "
            "install script."
        ),
    )
    parser.add_argument(
        "--with-cmake",
        dest="cmake_exe",
        metavar="EXE",
        required=False,
        default="cmake",
        help="Path to CMake executable (if not on PATH).",
    )
    parser.add_argument(
        "--cmake-generator",
        dest="cmake_generator",
        required=False,
        default="Unix Makefiles",
        choices=["Unix Makefiles", "Ninja"],
        help="The CMake makefiles generator",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=True,
        help="Clean before build, and pull latest Legion.",
    )
    parser.add_argument(
        "--extra",
        dest="extra_flags",
        action="append",
        required=False,
        default=[],
        help="Extra CMake flags.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        help="Number of threads used to compile.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose build output.",
    )
    parser.add_argument(
        "--with-thrust",
        dest="thrust_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("THRUST_PATH"),
        help="Path to Thrust installation directory. The required version of "
        "Thrust is " + required_thrust_version + " or compatible.  If not "
        "provided, Thrust will be installed automatically.",
    )
    parser.add_argument(
        "--legion-dir",
        dest="legion_dir",
        required=False,
        default=None,
        help="Path to an existing Legion build directory.",
    )
    parser.add_argument(
        "--legion-url",
        dest="legion_url",
        required=False,
        default="https://gitlab.com/StanfordLegion/legion.git",
        help="Legion git URL to build Legate with.",
    )
    parser.add_argument(
        "--legion-branch",
        dest="legion_branch",
        required=False,
        default="control_replication",
        help="Legion branch to build Legate with.",
    )
    args, unknown = parser.parse_known_args()

    try:
        subprocess.check_output([args.cmake_exe, "--version"])
    except OSError:
        print(
            "Error: CMake is not installed or otherwise not executable. "
            "Please check"
        )
        print(
            "your CMake installation and try again. You can use the "
            "--with-cmake flag"
        )
        print("to specify the CMake executable if it is not on PATH.")
        print()
        print("Attempted to execute: %s" % args.cmake_exe)
        sys.exit(1)

    install(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()
