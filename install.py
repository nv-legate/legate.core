#!/usr/bin/env python3

# Copyright 2021-2023 NVIDIA Corporation
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
import shutil
import subprocess
import sys
from distutils import sysconfig

# Flush output on newlines
sys.stdout.reconfigure(line_buffering=True)


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


def execute_command(args, verbose, ignore_errors=False, **kwargs):
    if verbose:
        print(f"Executing: {' '.join(args)} with {kwargs}")
    if ignore_errors:
        subprocess.call(args, **kwargs)
    else:
        subprocess.check_call(args, **kwargs)


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
    # Homebrew or pkg mgr installations may give bad values for LDLIBRARY.
    # Uses a fallback default path in case LDLIBRARY fails.
    default_libname = "libpython" + cv["LDVERSION"] + ".a"
    libdirs = [cv["LIBDIR"], cv["LIBPL"]]
    libnames = [cv["LDLIBRARY"], default_libname]
    paths = [
        os.path.join(libdir, libname)
        for libdir in libdirs
        for libname in libnames
    ]
    # ensure that static libraries are replaced with the dynamic version
    paths = [
        os.path.splitext(p)[0]
        + (".dylib" if platform.system() == "Darwin" else ".so")
        for p in paths
    ]
    paths = [p for p in paths if os.path.isfile(p)]
    e = "Error: could not auto-locate python library."
    assert paths, e
    return version, paths[0]


def scikit_build_cmake_build_dir(skbuild_dir):
    if os.path.exists(skbuild_dir):
        for f in os.listdir(skbuild_dir):
            if os.path.exists(
                cmake_build := os.path.join(skbuild_dir, f, "cmake-build")
            ):
                return cmake_build
    return None


def find_cmake_val(pattern, filepath):
    return (
        subprocess.check_output(["grep", "--color=never", pattern, filepath])
        .decode("UTF-8")
        .strip()
    )


def was_previously_built_with_different_build_isolation(
    isolated, legate_build_dir
):
    if (
        legate_build_dir is not None
        and os.path.exists(legate_build_dir)
        and os.path.exists(
            cmake_cache := os.path.join(legate_build_dir, "CMakeCache.txt")
        )
    ):
        try:
            if isolated:
                return True
            if find_cmake_val("pip-build-env", cmake_cache):
                return True
        except Exception:
            pass
    return False


def get_install_dir(args: list):
    # Infer the location where to install the Legion Python bindings,
    # otherwise they'll only be installed into the local scikit-build
    # cmake-install dir

    def get_arg(flag: str):
        idx = args.index(flag) if flag in args else -1
        if idx > -1 and idx + 1 < len(args):
            val = args[idx + 1]
            args.remove(flag)
            args.remove(val)
            return val
        return None

    install_dir = None
    install_args = []
    if root_dir := get_arg("--root"):
        install_dir = root_dir
        install_args += ["--root", root_dir]

    if prefix_dir := get_arg("--prefix"):
        install_dir = [install_dir, prefix_dir]
        install_dir = filter(lambda x: x, install_dir)
        install_dir = os.path.join(*install_dir, "")
        install_args += ["--prefix", prefix_dir]

    if install_dir and os.path.exists(install_dir):
        return (install_dir, install_args)

    # Install into conda or venv prefix if defined
    if "PREFIX" in os.environ and (os.environ.get("CONDA_BUILD", "0") == "1"):
        install_dir = os.environ["PREFIX"]
        return (install_dir, ["--root", "/", "--prefix", install_dir])
    elif "CONDA_PREFIX" in os.environ:
        install_dir = os.environ["CONDA_PREFIX"]
        return (install_dir, ["--root", "/", "--prefix", install_dir])
    elif "VIRTUAL_ENV" in os.environ:
        install_dir = os.environ["VIRTUAL_ENV"]
        return (install_dir, ["--root", "/", "--prefix", install_dir])

    import site

    # Try to install into user site packages first?
    if ("--user" in args or site.ENABLE_USER_SITE) and os.path.exists(
        user_base := site.getuserbase()
    ):
        if "--user" in args:
            args.remove("--user")
        return (user_base, ["--user"])

    # Otherwise fallback to regular site-packages?
    for site_pkgs in site.getsitepackages():
        install_dir = os.path.join(site_pkgs, "..", "..", "..")
        install_dir = os.path.realpath(install_dir)
        if install_dir != "/" and os.path.exists(install_dir):
            return (install_dir, ["--root", "/", "--prefix", install_dir])

    return (None, [])


def install_legion_python_bindings(
    verbose, cmake_exe, legate_build_dir, legion_dir, install_dir
):
    join = os.path.join
    exists = os.path.exists

    # Install Legion Python bindings if `legion_dir` is a Legion build dir
    # or if we built Legion as a side-effect of building `legate_core`
    if legion_dir is None or not exists(join(legion_dir, "CMakeCache.txt")):
        legion_dir = None
        if legate_build_dir and exists(legate_build_dir):
            if exists(
                legion_build_dir := join(
                    legate_build_dir, "_deps", "legion-build"
                )
            ):
                legion_dir = legion_build_dir

    if legion_dir is not None:
        if verbose:
            print(f"installing legion python bindings to {install_dir}")
        execute_command(
            [cmake_exe, "--install", join(legion_dir, "bindings", "python")]
            + ([] if not install_dir else ["--prefix", install_dir]),
            verbose,
        )


def install_legion_jupyter_notebook(
    verbose, cmake_exe, legate_build_dir, legion_dir, install_dir
):
    join = os.path.join
    exists = os.path.exists

    # Install Legion Jupyter Notebook if `legion_dir` is a Legion build dir
    # or if we built Legion as a side-effect of building `legate_core`
    if legion_dir is None or not exists(join(legion_dir, "CMakeCache.txt")):
        legion_dir = None
        if legate_build_dir and exists(legate_build_dir):
            if exists(
                legion_build_dir := join(
                    legate_build_dir, "_deps", "legion-build"
                )
            ):
                legion_dir = legion_build_dir

    if legion_dir is not None:
        if verbose:
            print(f"installing legion jupyter notebook to {install_dir}")
        execute_command(
            [cmake_exe, "--install", join(legion_dir, "jupyter_notebook")]
            + ([] if not install_dir else ["--prefix", install_dir]),
            verbose,
        )


def install(
    networks,
    cuda,
    arch,
    openmp,
    march,
    profiler,
    spy,
    build_docs,
    conduit,
    gasnet_system,
    nccl_dir,
    cmake_exe,
    cmake_generator,
    gasnet_dir,
    ucx_dir,
    cuda_dir,
    maxdim,
    maxfields,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    extra_flags,
    build_tests,
    build_examples,
    is_conda,
    editable,
    with_dependencies,
    build_isolation,
    thread_count,
    verbose,
    thrust_dir,
    legion_dir,
    legion_src_dir,
    legion_url,
    legion_branch,
    unknown,
):
    if len(networks) > 1:
        print(
            "Warning: Building Realm with multiple networking backends is not "
            "fully supported currently."
        )

    if clean_first is None:
        clean_first = not editable

    if legion_dir is not None and legion_src_dir is not None:
        sys.exit("Cannot specify both --legion-dir and --legion-src-dir")

    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    realpath = os.path.realpath

    cmake_exe = realpath(cmake_exe)

    print(f"Verbose build is {'on' if verbose else 'off'}")
    if verbose:
        print(f"networks: {networks}")
        print(f"cuda: {cuda}")
        print(f"arch: {arch}")
        print(f"openmp: {openmp}")
        print(f"march: {march}")
        print(f"profiler: {profiler}")
        print(f"spy: {spy}")
        print(f"build_docs: {build_docs}")
        print(f"conduit: {conduit}")
        print(f"gasnet_system: {gasnet_system}")
        print(f"nccl_dir: {nccl_dir}")
        print(f"cmake_exe: {cmake_exe}")
        print(f"cmake_generator: {cmake_generator}")
        print(f"gasnet_dir: {gasnet_dir}")
        print(f"ucx_dir: {ucx_dir}")
        print(f"cuda_dir: {cuda_dir}")
        print(f"maxdim: {maxdim}")
        print(f"maxfields: {maxfields}")
        print(f"debug: {debug}")
        print(f"debug_release: {debug_release}")
        print(f"check_bounds: {check_bounds}")
        print(f"clean_first: {clean_first}")
        print(f"extra_flags: {extra_flags}")
        print(f"editable: {editable}")
        print(f"with_dependencies: {with_dependencies}")
        print(f"build_isolation: {build_isolation}")
        print(f"thread_count: {thread_count}")
        print(f"verbose: {verbose}")
        print(f"thrust_dir: {thrust_dir}")
        print(f"legion_dir: {legion_dir}")
        print(f"legion_src_dir: {legion_src_dir}")
        print(f"legion_url: {legion_url}")
        print(f"legion_branch: {legion_branch}")
        print(f"unknown: {unknown}")

    legate_core_dir = dirname(realpath(__file__))

    pyversion, pylib_name = find_active_python_version_and_path()
    print(f"Using python lib and version: {pylib_name}, {pyversion}")

    def validate_path(path):
        if path is None or (path := str(path)) == "":
            return None
        if not os.path.isabs(path):
            path = join(legate_core_dir, path)
        if not exists(path := realpath(path)):
            print(f"Error: path does not exist: {path}")
            sys.exit(1)
        return path

    cuda_dir = validate_path(cuda_dir)
    nccl_dir = validate_path(nccl_dir)
    legion_dir = validate_path(legion_dir)
    legion_src_dir = validate_path(legion_src_dir)
    gasnet_dir = validate_path(gasnet_dir)
    ucx_dir = validate_path(ucx_dir)
    thrust_dir = validate_path(thrust_dir)

    if verbose:
        print(f"legate_core_dir: {legate_core_dir}")
        print(f"cuda_dir: {cuda_dir}")
        print(f"nccl_dir: {nccl_dir}")
        print(f"legion_dir: {legion_dir}")
        print(f"legion_src_dir: {legion_src_dir}")
        print(f"gasnet_dir: {gasnet_dir}")
        print(f"ucx_dir: {ucx_dir}")
        print(f"thrust_dir: {thrust_dir}")

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    skbuild_dir = join(legate_core_dir, "_skbuild")
    legate_build_dir = scikit_build_cmake_build_dir(skbuild_dir)

    if was_previously_built_with_different_build_isolation(
        build_isolation, legate_build_dir
    ):
        print("Performing a clean build to accommodate build isolation.")
        clean_first = True

    cmd_env = dict(os.environ.items())

    # Explicitly uninstall legate.core if doing a clean/isolated build.
    #
    # A prior installation may have built and installed legate.core C++
    # dependencies (like Legion).
    #
    # CMake will find and use them for the current build, which would normally
    # be correct, but pip uninstalls files from any existing installation as
    # the last step of the install process, including the libraries found by
    # CMake during the current build.
    #
    # Therefore this uninstall step must occur *before* CMake attempts to find
    # these dependencies, triggering CMake to build and install them again.
    if clean_first or build_isolation:
        execute_command(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "legate-core",
                "legion",
            ],
            verbose,
            ignore_errors=True,
            cwd=legate_core_dir,
            env=cmd_env,
        )

    if clean_first:
        shutil.rmtree(skbuild_dir, ignore_errors=True)
        shutil.rmtree(join(legate_core_dir, "dist"), ignore_errors=True)
        shutil.rmtree(join(legate_core_dir, "build"), ignore_errors=True)
        shutil.rmtree(
            join(legate_core_dir, "legate_core.egg-info"),
            ignore_errors=True,
        )

    # Configure and build legate.core via setup.py
    pip_install_cmd = [sys.executable, "-m", "pip", "install"]

    # Use preexisting CMAKE_ARGS from conda if set
    cmake_flags = cmd_env.get("CMAKE_ARGS", "").split(" ")

    if unknown is None:
        unknown = []

    install_dir, install_args = get_install_dir(unknown)

    pip_install_cmd += install_args
    pip_install_cmd += unknown

    if verbose:
        print(f"install_dir: {install_dir}")
        print("install_args:", install_args)

    if is_conda and "OPENSSL_DIR" not in os.environ:
        cmd_env.update({"OPENSSL_DIR": install_dir})

    if not with_dependencies:
        pip_install_cmd += ["--no-deps"]

    if not build_isolation:
        pip_install_cmd += ["--no-build-isolation"]

    pip_install_cmd += [f"--global-option=--cmake-executable={cmake_exe}"]

    if editable:
        cmd_env.update({"SETUPTOOLS_ENABLE_FEATURES": "legacy-editable"})
        cmake_flags += ["-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"]
        pip_install_cmd += ["--editable"]
    else:
        pip_install_cmd += ["--upgrade"]

    pip_install_cmd += ["."]

    if verbose:
        pip_install_cmd += ["-vv"]

    if debug or verbose:
        cmake_flags += [f"--log-level={'DEBUG' if debug else 'VERBOSE'}"]

    cmake_flags += f"""\
-DCMAKE_BUILD_TYPE={(
    "Debug" if debug else "RelWithDebInfo" if debug_release else "Release"
)}
-DBUILD_SHARED_LIBS=ON
-DLegion_MAX_DIM={str(maxdim)}
-DLegion_MAX_FIELDS={str(maxfields)}
-DLegion_SPY={("ON" if spy else "OFF")}
-DLegion_BOUNDS_CHECKS={("ON" if check_bounds else "OFF")}
-DLegion_USE_CUDA={("ON" if cuda else "OFF")}
-DLegion_USE_OpenMP={("ON" if openmp else "OFF")}
-DLegion_NETWORKS={";".join(networks)}
-DLegion_USE_Python=ON
-DLegion_Python_Version={pyversion}
-DLegion_BUILD_JUPYTER=ON
-DLegion_EMBED_GASNet_CONFIGURE_ARGS="--with-ibv-max-hcas=8"
""".splitlines()

    if march:
        cmake_flags += [f"-DBUILD_MARCH={march}"]
    if profiler:
        cmake_flags += ["-DLegion_BUILD_RUST_PROFILER=ON"]
    if cuda:
        cmake_flags += [f"-DLegion_CUDA_ARCH={arch}"]
    if nccl_dir:
        cmake_flags += [f"-DNCCL_DIR={nccl_dir}"]
    if gasnet_dir:
        cmake_flags += [f"-DGASNet_ROOT_DIR={gasnet_dir}"]
    if ucx_dir:
        cmake_flags += [f"-DUCX_ROOT={ucx_dir}"]
    if conduit:
        cmake_flags += [f"-DGASNet_CONDUIT={conduit}"]
    if gasnet_system:
        cmake_flags += [f"-DGASNet_SYSTEM={gasnet_system}"]
    if cuda_dir:
        cmake_flags += [f"-DCUDAToolkit_ROOT={cuda_dir}"]
    if thrust_dir:
        cmake_flags += [f"-DThrust_ROOT={thrust_dir}"]
    if legion_dir:
        cmake_flags += [f"-DLegion_ROOT={legion_dir}"]
    elif legion_src_dir:
        cmake_flags += [f"-DCPM_Legion_SOURCE={legion_src_dir}"]
    else:
        cmake_flags += ["-DCPM_DOWNLOAD_Legion=ON"]
    if legion_url:
        cmake_flags += [f"-Dlegate_core_LEGION_REPOSITORY={legion_url}"]
    if legion_branch:
        cmake_flags += [f"-Dlegate_core_LEGION_BRANCH={legion_branch}"]
    if build_docs:
        cmake_flags += ["-Dlegate_core_BUILD_DOCS=ON"]
    if build_tests:
        cmake_flags += ["-Dlegate_core_BUILD_TESTS=ON"]
    if build_examples:
        cmake_flags += ["-Dlegate_core_BUILD_EXAMPLES=ON"]

    if not is_conda and editable:
        # If editable, install the Legion python bindings
        # into scikit-build's "cmake-install" directory,
        # not into the real `install_dir`
        install_args = ["--root", "/", "--prefix", "${CMAKE_INSTALL_PREFIX}"]

    if len(install_args) > 0:
        cmake_flags += [
            f'-DLegion_PYTHON_EXTRA_INSTALL_ARGS="{";".join(install_args)}"'
        ]

    cmake_flags += extra_flags
    build_flags = [f"-j{str(thread_count)}"]
    if verbose:
        if cmake_generator == "Unix Makefiles":
            build_flags += ["VERBOSE=1"]
        else:
            build_flags += ["--verbose"]

    cmd_env.update(
        {
            "CMAKE_ARGS": " ".join(cmake_flags),
            "CMAKE_GENERATOR": cmake_generator,
            "SKBUILD_BUILD_OPTIONS": " ".join(build_flags),
        }
    )

    # execute python -m pip install <args> .
    execute_command(pip_install_cmd, verbose, cwd=legate_core_dir, env=cmd_env)

    install_legion_python_bindings(
        verbose,
        cmake_exe,
        legate_build_dir,
        legion_dir,
        None if editable else install_dir,
    )
    install_legion_jupyter_notebook(
        verbose,
        cmake_exe,
        legate_build_dir,
        legion_dir,
        None if editable else install_dir,
    )


def driver():
    is_conda = ("CONDA_PREFIX" in os.environ) or (
        "PREFIX" in os.environ and os.environ.get("CONDA_BUILD", "0") == "1"
    )

    parser = argparse.ArgumentParser(description="Install Legate front end.")
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
        "--network",
        dest="networks",
        action="append",
        required=False,
        choices=["gasnet1", "gasnetex", "ucx", "mpi"],
        default=[],
        help="Realm networking backend to use for multi-node execution.",
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
        "--with-ucx",
        dest="ucx_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("UCX_ROOT"),
        help="Path to UCX installation directory.",
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
        default="all-major",
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
        default=("haswell" if platform.machine() == "x86_64" else None),
        help="Specify the target CPU architecture.",
    )
    parser.add_argument(
        "--profiler",
        dest="profiler",
        action=BooleanFlag,
        default=False,
        help="Build Rust version of Legion profiler.",
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
        "--docs",
        dest="build_docs",
        action="store_true",
        required=False,
        default=False,
        help="Build Doxygen docs.",
    )
    parser.add_argument(
        "--conduit",
        dest="conduit",
        action="store",
        required=False,
        # TODO: To support UDP conduit, we would need to add a special case on
        # the legate launcher.
        # See https://github.com/nv-legate/legate.core/issues/294.
        choices=["ibv", "ucx", "aries", "mpi", "ofi"],
        default=os.environ.get("CONDUIT"),
        help="Build Legate with specified GASNet conduit.",
    )
    parser.add_argument(
        "--gasnet-system",
        dest="gasnet_system",
        action="store",
        required=False,
        default=None,
        help="Specify a system-specific configuration to use for GASNet",
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
        default=os.environ.get(
            "CMAKE_GENERATOR",
            "Unix Makefiles" if shutil.which("ninja") is None else "Ninja",
        ),
        choices=["Ninja", "Unix Makefiles", None],
        help="The CMake makefiles generator",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=None,
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
        "--build-tests",
        dest="build_tests",
        action=BooleanFlag,
        default=False,
        help="Whether to build the tests",
    )
    parser.add_argument(
        "--build-examples",
        dest="build_examples",
        action=BooleanFlag,
        default=False,
        help="Whether to build the examples",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        default=os.environ.get("CPU_COUNT"),
        help="Number of threads used to compile.",
    )
    parser.add_argument(
        "--editable",
        dest="editable",
        action="store_true",
        required=False,
        default=False,
        help="Perform an editable install. Defaults to "
        "--no-clean unless --clean is passed explicitly.",
    )
    parser.add_argument(
        "--deps",
        dest="with_dependencies",
        action=BooleanFlag,
        required=False,
        default=(not is_conda),
        help="Don't install package dependencies.",
    )
    parser.add_argument(
        "--build-isolation",
        dest="build_isolation",
        action=BooleanFlag,
        required=False,
        default=(not is_conda),
        help="Enable isolation when building a modern source distribution. "
        "Build dependencies specified by PEP 518 must be already "
        "installed if this option is used.",
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
        "Thrust is cuda-11.2 or compatible.  If not "
        "provided, Thrust will be installed automatically.",
    )
    parser.add_argument(
        "--with-legion",
        dest="legion_dir",
        required=False,
        default=None,
        help="Path to an existing Legion build directory.",
    )
    parser.add_argument(
        "--legion-src-dir",
        dest="legion_src_dir",
        required=False,
        default=None,
        help="Path to an existing Legion source directory.",
    )
    parser.add_argument(
        "--legion-url",
        dest="legion_url",
        required=False,
        default=None,
        help="Legion git URL to build Legate with.",
    )
    parser.add_argument(
        "--legion-branch",
        dest="legion_branch",
        required=False,
        default=None,
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
        print(f"Attempted to execute: {args.cmake_exe}")
        sys.exit(1)

    install(unknown=unknown, is_conda=is_conda, **vars(args))


if __name__ == "__main__":
    driver()
