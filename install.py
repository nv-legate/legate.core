#!/usr/bin/env python

# Copyright 2021 NVIDIA Corporation
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
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from distutils import sysconfig

_version = sys.version_info.major

try:
    _input = raw_input  # Python 2.x:
except NameError:
    _input = input  # Python 3.x:

# reopen stdout file descriptor with write mode
# and 0 as the buffer size (unbuffered)
# import io

try:
    # Python 3, open as binary, then wrap in a TextIOWrapper with
    # write-through.
    #
    # sys.stdout =
    # io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
    #
    # If flushing on newlines is sufficient, as of 3.7 you can instead just
    # call:
    sys.stdout.reconfigure(line_buffering=True)
except TypeError:
    # Python 2
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 0)

os_name = platform.system()

required_thrust_version = "cuda-11.2"

# Global variable for verbose installation
verbose_global = False


def verbose_check_call(*args, **kwargs):
    if verbose_global:
        print('Executing: "', " ".join(*args), '" with ', kwargs)
    subprocess.check_call(*args, **kwargs)


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
        p.replace(".a", ".dylib" if os_name == "Darwin" else ".so")
        for p in paths
    ]
    paths = [p for p in paths if os.path.isfile(p)]
    e = "Error: could not auto-locate python library."
    assert paths, e
    return version, paths[0]


def git_clone(repo_dir, url, branch=None, tag=None, commit=None):
    assert branch is not None or tag is not None or commit is not None
    if branch is not None:
        verbose_check_call(
            ["git", "clone", "--recursive", "-b", branch, url, repo_dir]
        )
    elif commit is not None:
        verbose_check_call(["git", "clone", "--recursive", url, repo_dir])
        verbose_check_call(["git", "checkout", commit], cwd=repo_dir)
        verbose_check_call(
            ["git", "submodule", "update", "--init"], cwd=repo_dir
        )
    else:
        verbose_check_call(
            [
                "git",
                "clone",
                "--recursive",
                "--single-branch",
                "-b",
                tag,
                url,
                repo_dir,
            ]
        )
        verbose_check_call(["git", "checkout", "-b", "master"], cwd=repo_dir)


def git_reset(repo_dir, refspec):
    verbose_check_call(["git", "reset", "--hard", refspec], cwd=repo_dir)


def git_update(repo_dir, branch=None):
    verbose_check_call(["git", "pull", "--ff-only"], cwd=repo_dir)
    if branch is not None:
        verbose_check_call(["git", "checkout", branch], cwd=repo_dir)


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


def dump_json_config(filename, value):
    with open(filename, "w") as f:
        return json.dump(value, f)


def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)


def install_gasnet(gasnet_dir, conduit, thread_count):
    print("Legate is installing GASNet into a local directory...")
    temp_dir = tempfile.mkdtemp()
    git_clone(
        temp_dir,
        url="https://github.com/StanfordLegion/gasnet.git",
        branch="master",
    )
    # Update the configuration file with the prefix for our output
    # Then we can invoke make
    verbose_check_call(
        [
            "make",
            "-j",
            str(thread_count),
            "CONDUIT=" + str(conduit),
            "GASNET_INSTALL_DIR=" + str(gasnet_dir),
        ],
        cwd=temp_dir,
    )
    shutil.rmtree(temp_dir)


def install_legion(legion_src_dir, branch="legate_stable"):
    print("Legate is installing Legion into a local directory...")
    # For now all we have to do is clone legion since we build it with Legate
    git_clone(
        legion_src_dir,
        url="https://gitlab.com/StanfordLegion/legion.git",
        branch=branch,
    )


def install_thrust(thrust_dir):
    print("Legate is installing Thrust into a local directory...")
    git_clone(
        thrust_dir,
        url="https://github.com/thrust/thrust.git",
        tag=required_thrust_version,
    )


def update_legion(legion_src_dir, branch="legate_stable"):
    # Make sure we are on the right branch for single/multi-node
    git_update(legion_src_dir, branch=branch)


def build_legion(
    legion_src_dir,
    install_dir,
    cmake,
    cmake_exe,
    cuda_dir,
    debug,
    debug_release,
    check_bounds,
    cuda,
    arch,
    openmp,
    llvm,
    hdf,
    spy,
    gasnet,
    gasnet_dir,
    conduit,
    no_hijack,
    pyversion,
    pylib_name,
    maxdim,
    maxfields,
    clean_first,
    extra_flags,
    thread_count,
    verbose,
):
    if no_hijack and cmake:
        print(
            "Warning: CMake build does not support no-hijack mode. Falling "
            "back to GNU make build."
        )
        cmake = False

    if cmake:
        build_dir = os.path.join(legion_src_dir, "build")
        if clean_first:
            try:
                shutil.rmtree(build_dir)
            except FileNotFoundError:
                pass
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        flags = (
            [
                "-DCMAKE_BUILD_TYPE=%s"
                % (
                    "Debug"
                    if debug
                    else "RelWithDebInfo"
                    if debug_release
                    else "Release"
                ),
                "-DLegion_MAX_DIM=%s" % (str(maxdim)),
                "-DLegion_MAX_FIELDS=%s" % (str(maxfields)),
                "-DLegion_USE_CUDA=%s" % ("ON" if cuda else "OFF"),
                "-DLegion_GPU_ARCH=%s" % arch,
                "-DLegion_USE_OpenMP=%s" % ("ON" if openmp else "OFF"),
                "-DLegion_USE_LLVM=%s" % ("ON" if llvm else "OFF"),
                "-DLegion_USE_GASNet=%s" % ("ON" if gasnet else "OFF"),
                "-DLegion_USE_HDF5=%s" % ("ON" if hdf else "OFF"),
                "-DCMAKE_INSTALL_PREFIX=%s" % (os.path.realpath(install_dir)),
                "-DLegion_USE_Python=On",
                "-DLegion_Python_Version=%s" % pyversion,
                "-DLegion_REDOP_COMPLEX=On",
                "-DLegion_REDOP_HALF=On",
                "-DBUILD_SHARED_LIBS=ON",
                "-DLegion_BUILD_BINDINGS=On",
            ]
            + extra_flags
            + (["-DLegion_BOUNDS_CHECKS=On"] if check_bounds else [])
            + (["-DLegion_HIJACK_CUDART=Off"] if no_hijack else [])
            + (
                ["-DGASNet_ROOT_DIR=%s" % gasnet_dir]
                if gasnet_dir is not None
                else []
            )
            + (
                ["-DGASNet_CONDUIT=%s" % conduit]
                if conduit is not None
                else []
            )
            + (
                ["-DCUDA_TOOLKIT_ROOT_DIR=%s" % cuda_dir]
                if cuda_dir is not None
                else []
            )
            + (
                ["-DCMAKE_CXX_COMPILER=%s" % os.environ["CXX"]]
                if "CXX" in os.environ
                else []
            )
            + (
                ["-DCMAKE_CXX_FLAGS=%s" % os.environ["CC_FLAGS"]]
                if "CC_FLAGS" in os.environ
                else []
            )
        )
        make_flags = ["VERBOSE=1"] if verbose else []
        make_flags += ["-C", os.path.realpath(build_dir)]
        if spy:
            raise NotImplementedError("Need support for Legion Spy with cmake")
        try:
            subprocess.check_output([cmake_exe, "--version"])
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
            print("Attempted to execute: %s" % cmake_exe)
            sys.exit(1)
        verbose_check_call(
            [cmake_exe] + flags + [legion_src_dir], cwd=build_dir
        )
        verbose_check_call(
            ["make"] + make_flags + ["-j", str(thread_count), "install"],
            cwd=build_dir,
        )
        # TODO: install legion spy and legion prof
    else:
        version = pyversion.split(".")
        flags = (
            [
                "LG_RT_DIR=%s" % (os.path.join(legion_src_dir, "runtime")),
                "DEBUG=%s" % (1 if debug else 0),
                "DEBUG_RELEASE=%s" % (1 if debug_release else 0),
                "MAX_DIM=%s" % (str(maxdim)),
                "MAX_FIELDS=%s" % (str(maxfields)),
                "USE_CUDA=%s" % (1 if cuda else 0),
                "GPU_ARCH=%s" % arch,
                "USE_OPENMP=%s" % (1 if openmp else 0),
                "USE_LLVM=%s" % (1 if llvm else 0),
                "USE_GASNET=%s" % (1 if gasnet else 0),
                "USE_HDF=%s" % (1 if hdf else 0),
                "PREFIX=%s" % (os.path.realpath(install_dir)),
                "PYTHON_VERSION_MAJOR=%s" % version[0],
                "PYTHON_VERSION_MINOR=%s" % version[1],
                "PYTHON_LIB=%s" % pylib_name,
                "FORCE_PYTHON=1",
                "USE_COMPLEX=1",
                "USE_HALF=1",
                "USE_SPY=%s" % (1 if spy else 0),
                "REALM_USE_CUDART_HIJACK=%s" % (1 if not no_hijack else 0),
            ]
            + extra_flags
            + (["BOUNDS_CHECKS=1"] if check_bounds else [])
            + (["GASNET=%s" % gasnet_dir] if gasnet_dir is not None else [])
            + (["CONDUIT=%s" % conduit] if conduit is not None else [])
            + (["CUDA=%s" % cuda_dir] if cuda_dir is not None else [])
        )

        legion_python_dir = os.path.join(legion_src_dir, "bindings", "python")
        if clean_first:
            verbose_check_call(
                ["make"] + flags + ["clean"], cwd=legion_python_dir
            )
        verbose_check_call(
            ["make"] + flags + ["-j", str(thread_count), "install"],
            cwd=legion_python_dir,
        )
        verbose_check_call(
            [
                sys.executable,
                "setup.py",
                "install",
                "--prefix",
                str(os.path.realpath(install_dir)),
            ],
            cwd=legion_python_dir,
        )
    verbose_check_call(
        [
            "cp",
            "legion_spy.py",
            os.path.join(install_dir, "share", "legate", "legion_spy.py"),
        ],
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    verbose_check_call(
        [
            "cp",
            "legion_prof.py",
            os.path.join(install_dir, "share", "legate", "legion_prof.py"),
        ],
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    verbose_check_call(
        [
            "cp",
            "legion_serializer.py",
            os.path.join(
                install_dir, "share", "legate", "legion_serializer.py"
            ),
        ],
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    verbose_check_call(
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
        cwd=os.path.join(legion_src_dir, "tools"),
    )
    verbose_check_call(
        [
            "cp",
            "-r",
            "legion_prof_files",
            os.path.join(install_dir, "share", "legate", "legion_prof_files"),
        ],
        cwd=os.path.join(legion_src_dir, "tools"),
    )


def build_legate_core(
    install_dir,
    legate_dir,
    cmake,
    cmake_exe,
    cuda_dir,
    debug,
    debug_release,
    cuda,
    arch,
    openmp,
    spy,
    gasnet,
    clean_first,
    thread_count,
    verbose,
    unknown,
):
    src_dir = os.path.join(legate_dir, "src")
    if cmake:
        print("Warning: CMake is currently not supported for Legate build.")
        print("Using GNU Make for now.")

    make_flags = [
        "LEGATE_DIR=%s" % install_dir,
        "DEBUG=%s" % (1 if debug else 0),
        "DEBUG_RELEASE=%s" % (1 if debug_release else 0),
        "USE_CUDA=%s" % (1 if cuda else 0),
        "GPU_ARCH=%s" % arch,
        "PREFIX=%s" % str(install_dir),
        "USE_GASNET=%s" % (1 if gasnet else 0),
    ] + (["CUDA=%s" % cuda_dir] if cuda_dir is not None else [])
    if clean_first:
        verbose_check_call(["make"] + make_flags + ["clean"], cwd=src_dir)
    verbose_check_call(
        ["make"] + make_flags + ["-j", str(thread_count), "install"],
        cwd=src_dir,
    )
    # Fill in config.mk.in and copy it to the target destination
    with open(os.path.join(src_dir, "config.mk.in")) as f:
        content = f.read()
    content = content.format(
        debug=repr(1 if debug else 0),
        debug_release=repr(1 if debug_release else 0),
        cuda=repr(1 if cuda else 0),
        arch=(arch if arch is not None else ""),
        cudadir=(cuda_dir if cuda_dir is not None else ""),
        openmp=repr(1 if openmp else 0),
        gasnet=repr(1 if gasnet else 0),
    )
    with open(os.path.join(src_dir, "config.mk"), "wb") as f:
        f.write(content.encode("utf-8"))
    cmd = ["cp", "config.mk", os.path.join(install_dir, "share", "legate")]
    verbose_check_call(cmd, cwd=src_dir)
    # Then run setup.py
    cmd = [sys.executable, "setup.py", "install", "--recurse"]
    if unknown is not None:
        try:
            prefix_loc = unknown.index("--prefix")
            cmd.extend(unknown[prefix_loc : prefix_loc + 2])
        except ValueError:
            cmd += ["--prefix", str(install_dir)]
    else:
        cmd += ["--prefix", str(install_dir)]
    verbose_check_call(cmd, cwd=legate_dir)


def get_cmake_config(cmake, legate_dir, default=None):
    config_filename = os.path.join(legate_dir, ".cmake.json")
    if cmake is None:
        cmake = load_json_config(config_filename)
        if cmake is None:
            cmake = default
    assert cmake in [True, False]
    dump_json_config(config_filename, cmake)
    return cmake


def install(
    gasnet=False,
    cuda=False,
    arch=None,
    openmp=False,
    hdf=False,
    llvm=False,
    spy=False,
    conduit=None,
    no_hijack=True,
    cmake=None,
    cmake_exe=None,
    install_dir=None,
    gasnet_dir=None,
    legion_dir=None,
    pylib_name=None,
    cuda_dir=None,
    maxdim=3,
    maxfields=256,
    debug=False,
    debug_release=False,
    check_bounds=False,
    clean_first=None,
    extra_flags=[],
    thread_count=None,
    verbose=False,
    thrust_dir=None,
    legion_branch=None,
    unknown=None,
):
    global verbose_global
    verbose_global = verbose

    legate_dir = os.path.dirname(os.path.realpath(__file__))

    cmake = get_cmake_config(cmake, legate_dir, default=False)

    if clean_first is None:
        clean_first = not cmake

    if pylib_name is None:
        pyversion, pylib_name = find_active_python_version_and_path()
    else:
        f_name = os.path.split(pylib_name)[-1]
        match = re.match(r"^libpython(\d\d?\.\d\d?)", f_name)
        e = "Unable to get version from library name {}".format(pylib_name)
        assert match, e
        pyversion = match.group(1)
    print("Using python lib and version: {}, {}".format(pylib_name, pyversion))

    install_dir_config = os.path.join(legate_dir, ".install-dir.json")
    if install_dir is None:
        install_dir = load_json_config(install_dir_config)
        if install_dir is None:
            install_dir = os.path.join(legate_dir, "install")
    install_dir = os.path.realpath(install_dir)
    dump_json_config(install_dir_config, install_dir)
    os.makedirs(os.path.join(install_dir, "share", "legate"), exist_ok=True)

    thread_count = thread_count
    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Save the maxdim config
    maxdim_config = os.path.join(legate_dir, ".maxdim.json")
    if "LEGION_MAX_DIM" in os.environ:
        # Subtract one to get to the legate max dim
        maxdim = int(os.environ["LEGION_MAX_DIM"]) - 1
    # Check the max dimensions
    # Legion could actually go up to 9 dimensions, but we leave an extra
    # free dimension for libraries to use as a free dimension
    if maxdim < 1 or maxdim > 8:
        raise Exception(
            "The maximum number of Legate dimensions must be between 1 and 8 "
            "inclusive"
        )
    # Convert back to legion max dimensions
    maxdim += 1
    dump_json_config(maxdim_config, str(maxdim))

    # Save the maxfields config
    maxfields_config = os.path.join(legate_dir, ".maxfields.json")
    if "LEGION_MAX_FIELDS" in os.environ:
        maxfields = int(os.environ["LEGION_MAX_FIELDS"])
    # Check that max fields is between 32 and 4096 and is a power of 2
    if maxfields not in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        raise Exception(
            "The maximum number of Legate fields must be a power of 2 between "
            "32 and 4096 inclusive"
        )
    dump_json_config(maxfields_config, str(maxfields))

    # If the user asked for a conduit and we don't have gasnet then install it
    if gasnet:
        conduit_config = os.path.join(legate_dir, ".conduit.json")
        if "CONDUIT" in os.environ:
            conduit = os.environ["CONDUIT"]
        else:
            if conduit is None:
                conduit = load_json_config(conduit_config)
                if conduit is None:
                    raise Exception(
                        "The first time you use GASNet you need to "
                        'tell us which conduit to use with the "--conduit" '
                        "flag."
                    )
        dump_json_config(conduit_config, conduit)
        gasnet_config = os.path.join(
            legate_dir, ".gasnet" + str(conduit) + ".json"
        )
        if "GASNET" in os.environ:
            gasnet_dir = os.environ["GASNET"]
        else:
            if gasnet_dir is None:
                gasnet_dir = load_json_config(gasnet_config)
                if gasnet_dir is None:
                    gasnet_dir = os.path.join(install_dir, "gasnet")
        if not os.path.exists(gasnet_dir):
            install_gasnet(gasnet_dir, conduit, thread_count)
        dump_json_config(gasnet_config, gasnet_dir)

    # If the user asked for CUDA, make sure we know where the install
    # directory is
    if cuda:
        cuda_config = os.path.join(legate_dir, ".cuda.json")
        # See if we can get it from the environment variables
        if "CUDA" in os.environ:
            cuda_dir = os.environ["CUDA"]
        else:
            if cuda_dir is None:
                cuda_dir = load_json_config(cuda_config)
                if cuda_dir is None:
                    raise Exception(
                        "The first time you use CUDA you need to tell Legate "
                        'where CUDA is installed with the "--with-cuda" flag.'
                    )
        dump_json_config(cuda_config, cuda_dir)

    # install a stable version of Thrust
    thrust_config = os.path.join(legate_dir, ".thrust.json")
    if "THRUST_PATH" in os.environ:
        thrust_dir = os.environ["THRUST_PATH"]
    elif thrust_dir is None:
        thrust_dir = load_json_config(thrust_config)
        if thrust_dir is None:
            thrust_dir = os.path.join(install_dir, "thrust")
    if not os.path.exists(thrust_dir):
        install_thrust(thrust_dir)
    else:
        thrust_dir = os.path.realpath(thrust_dir)
    # Simply put Thrust into the environment.
    os.environ["CXXFLAGS"] = (
        "-I" + thrust_dir + " " + os.environ.get("CXXFLAGS", "")
    )
    dump_json_config(thrust_config, thrust_dir)

    # Grab LEGION_DIR from the environment if available, otherwise
    # assume we're running relative to our own location.
    if legion_dir is None:
        if "LEGION_DIR" in os.environ:
            found_legion_install = True
            legion_dir = os.path.realpath(os.environ["LEGION_DIR"])
            legion_src_dir = None
        else:
            found_legion_install = False
            legion_src_dir = os.path.join(legate_dir, "legion")
            legion_dir = install_dir
    else:
        found_legion_install = True

    if not found_legion_install:
        # Check to see if Legion is up-to-date or get it if it isn't
        if os.path.exists(legion_src_dir):
            if clean_first:
                # Don't update Legion if not doing a clean build, to avoid
                # spurious build errors.
                update_legion(legion_src_dir, branch=legion_branch)
        else:
            install_legion(legion_src_dir, branch=legion_branch)

        build_legion(
            legion_src_dir,
            legion_dir,
            cmake,
            cmake_exe,
            cuda_dir,
            debug,
            debug_release,
            check_bounds,
            cuda,
            arch,
            openmp,
            llvm,
            hdf,
            spy,
            gasnet,
            gasnet_dir,
            conduit,
            no_hijack,
            pyversion,
            pylib_name,
            maxdim,
            maxfields,
            clean_first,
            extra_flags,
            thread_count,
            verbose,
        )

    build_legate_core(
        install_dir,
        legate_dir,
        cmake,
        cmake_exe,
        cuda_dir,
        debug,
        debug_release,
        cuda,
        arch,
        openmp,
        spy,
        gasnet,
        clean_first,
        thread_count,
        verbose,
        unknown,
    )
    # Copy any executables that we need for legate functionality
    verbose_check_call(
        ["cp", "legate.py", os.path.join(install_dir, "bin", "legate")],
        cwd=legate_dir,
    )
    if cuda:
        # Copy CUDA configuration that the launcher needs to find CUDA path
        verbose_check_call(
            [
                "cp",
                ".cuda.json",
                os.path.join(install_dir, "share", "legate", ".cuda.json"),
            ],
            cwd=legate_dir,
        )
    # Copy thrust configuration
    verbose_check_call(
        [
            "cp",
            thrust_config,
            os.path.join(install_dir, "share", "legate"),
        ],
        cwd=legate_dir,
    )


def driver():
    parser = argparse.ArgumentParser(description="Install Legate front end.")
    parser.add_argument(
        "--install-dir",
        dest="install_dir",
        metavar="DIR",
        required=False,
        help="Path to install all Legate-related software",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG") == "1",
        help="Build Legate with debugging enabled.",
    )
    parser.add_argument(
        "--debug-release",
        dest="debug_release",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG_RELEASE") == "1",
        help="Build Legate with debugging symbols enabled.",
    )
    parser.add_argument(
        "--check-bounds",
        dest="check_bounds",
        action="store_true",
        required=False,
        default=os.environ.get("CHECK_BOUNDS") == "1",
        help="Build Legate with bounds checkin enabled (warning: expensive).",
    )
    parser.add_argument(
        "--max-dim",
        dest="maxdim",
        type=int,
        default=3,
        help="Maximum number of dimensions that Legate will support",
    )
    parser.add_argument(
        "--max-fields",
        dest="maxfields",
        type=int,
        default=256,
        help="Maximum number of fields that Legate will support",
    )
    parser.add_argument(
        "--gasnet",
        dest="gasnet",
        action="store_true",
        required=False,
        default=os.environ.get("USE_GASNET") == "1",
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
        "--with-legion",
        dest="legion_dir",
        metavar="DIR",
        required=False,
        help="Path to Legion installation directory.",
    )
    parser.add_argument(
        "--cuda",
        dest="cuda",
        action="store_true",
        required=False,
        default=os.environ.get("USE_CUDA") == "1",
        help="Build Legate with CUDA.",
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
        default="volta",
        help="Specify the target GPU architecture.",
    )
    parser.add_argument(
        "--openmp",
        dest="openmp",
        action="store_true",
        required=False,
        default=os.environ.get("USE_OPENMP") == "1",
        help="Build Legate with OpenMP support.",
    )
    parser.add_argument(
        "--llvm",
        dest="llvm",
        action="store_true",
        required=False,
        default=os.environ.get("USE_LLVM") == "1",
        help="Build Legate with LLVM support.",
    )
    parser.add_argument(
        "--hdf5",
        "--hdf",
        dest="hdf",
        action="store_true",
        required=False,
        default=os.environ.get("USE_HDF") == "1",
        help="Build Legate with HDF.",
    )
    parser.add_argument(
        "--spy",
        dest="spy",
        action="store_true",
        required=False,
        default=os.environ.get("USE_SPY") == "1",
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
        "--with-hijack",
        dest="no_hijack",
        action="store_false",
        required=False,
        default=True,
        help=(
            "Activate the CUDA hijack in Realm "
            "(incompatible with Legate Pandas)."
        ),
    )
    parser.add_argument(
        "--python-lib",
        dest="pylib_name",
        action="store",
        required=False,
        default=None,
        help="Build Legate for the specified Python library.",
    )
    parser.add_argument(
        "--cmake",
        dest="cmake",
        action="store_true",
        required=False,
        default=None,
        help="Build Legate with CMake.",
    )
    parser.add_argument(
        "--no-cmake",
        dest="cmake",
        action="store_false",
        required=False,
        help="Don't build Legate with CMake (instead use GNU Make).",
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
        "--no-clean",
        "--noclean",
        dest="clean_first",
        action="store_false",
        required=False,
        default=True,
        help="Skip clean before build, and don't pull latest Legion.",
    )
    parser.add_argument(
        "--extra",
        dest="extra_flags",
        action="append",
        required=False,
        default=[],
        help="Extra flags for make command.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        help="Number threads used to compile.",
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
        help="Path to Thrust installation directory. The required version of "
        "Thrust is " + required_thrust_version + " or compatible.  If not "
        "provided, Thrust will be installed automatically.",
    )
    parser.add_argument(
        "--legion-branch",
        dest="legion_branch",
        action="store",
        required=False,
        default="legate_stable",
        help="Legion branch to build Legate with.",
    )
    args, unknown = parser.parse_known_args()
    install(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()
