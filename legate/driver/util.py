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
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Type, TypeVar

from .types import DataclassProtocol, LegatePaths, LegionPaths

__all__ = (
    "get_legate_build_dir",
    "get_legate_paths",
    "get_legion_paths",
    "object_to_dataclass",
    "read_c_define",
    "read_cmake_cache_value",
)


T = TypeVar("T", bound=DataclassProtocol)


def object_to_dataclass(obj: object, typ: Type[T]) -> T:
    kws = {name: getattr(obj, name) for name in typ.__dataclass_fields__}
    return typ(**kws)


def read_c_define(header_path: Path, name: str) -> str | None:
    try:
        with open(header_path, "r") as f:
            lines = (line for line in f if line.startswith("#define"))
            for line in lines:
                tokens = line.split(" ")
                if tokens[1].strip() == name:
                    return tokens[2].strip()
    except IOError:
        pass

    return None


def read_cmake_cache_value(pattern: str, file_path: Path) -> str:
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if re.match(pattern, line):
                return line.strip().split("=")[1]

    raise RuntimeError(f"Could not find value for {pattern} in {file_path}")


def get_legate_build_dir(legate_dir: Path) -> Path | None:
    # If using a local non-scikit-build CMake build dir, read
    # Legion_BINARY_DIR and Legion_SOURCE_DIR from CMakeCache.txt
    legate_build_dir = legate_dir / "build"
    cmake_cache_txt = legate_build_dir.joinpath("CMakeCache.txt")
    if legate_build_dir.exists() and cmake_cache_txt.exists():
        return legate_build_dir

    skbuild_dir = legate_dir / "_skbuild"
    if not skbuild_dir.exists():
        return None

    for f in skbuild_dir.iterdir():

        # If using a local scikit-build dir at _skbuild/<arch>/cmake-build,
        # read Legion_BINARY_DIR and Legion_SOURCE_DIR from CMakeCache.txt

        legate_build_dir = skbuild_dir / f / "cmake-build"
        cmake_cache_txt = legate_build_dir / "CMakeCache.txt"

        if legate_build_dir.exists() and cmake_cache_txt.exists():
            try:
                # Test whether FIND_LEGATE_CORE_CPP is set to ON. If it
                # isn't, then we built legate_core C++ as a side-effect of
                # building legate_core_python.
                read_cmake_cache_value(
                    "FIND_LEGATE_CORE_CPP:BOOL=OFF", cmake_cache_txt
                )
            except Exception:
                # If FIND_LEGATE_CORE_CPP is set to ON, check to see if
                # legate_core_DIR is a valid path. If it is, check whether
                # legate_core_DIR is a path to a legate_core build dir i.e.
                # `-D legate_core_ROOT=/legate.core/build`
                legate_core_dir = Path(
                    read_cmake_cache_value(
                        "legate_core_DIR:PATH=", cmake_cache_txt
                    )
                )

                # If legate_core_dir doesn't have a CMakeCache.txt, CMake's
                # find_package found a system legate_core installation.
                # Return the installation paths.
                cmake_cache_txt = legate_core_dir / "CMakeCache.txt"
                if cmake_cache_txt.exists():
                    return Path(
                        read_cmake_cache_value(
                            "legate_core_BINARY_DIR:STATIC=", cmake_cache_txt
                        )
                    )
                return None

            return legate_build_dir

    return None


def get_legate_paths() -> LegatePaths:
    import legate

    legate_dir = Path(legate.__path__[0]).parent
    legate_build_dir = get_legate_build_dir(legate_dir)

    if legate_build_dir is None:
        return LegatePaths(
            legate_dir=legate_dir,
            legate_build_dir=legate_build_dir,
            bind_sh_path=Path(sys.argv[0]).parent / "bind.sh",
            legate_lib_path=Path(sys.argv[0]).parents[1] / "lib",
        )

    cmake_cache_txt = legate_build_dir.joinpath("CMakeCache.txt")

    legate_source_dir = Path(
        read_cmake_cache_value(
            "legate_core_SOURCE_DIR:STATIC=", cmake_cache_txt
        )
    )

    legate_binary_dir = Path(
        read_cmake_cache_value(
            "legate_core_BINARY_DIR:STATIC=", cmake_cache_txt
        )
    )

    return LegatePaths(
        legate_dir=legate_dir,
        legate_build_dir=legate_build_dir,
        bind_sh_path=legate_source_dir / "bind.sh",
        legate_lib_path=legate_binary_dir / "lib",
    )


def get_legion_paths(legate_paths: LegatePaths) -> LegionPaths:

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

    def installed_legion_paths(
        legion_dir: Path, legion_module: Path | None = None
    ) -> LegionPaths:
        if legion_module is None:
            legion_lib_dir = legion_dir / "lib"
            for f in legion_lib_dir.iterdir():
                if legion_lib_dir.joinpath(f / "site-packages").exists():
                    legion_module = legion_lib_dir / f / "site-packages"
                    break

            legion_bin_path = legion_dir / "bin"
            legion_include_path = legion_dir / "include"

            return LegionPaths(
                legion_bin_path=legion_bin_path,
                legion_lib_path=legion_lib_dir,
                realm_defines_h=legion_include_path / "realm_defines.h",
                legion_defines_h=legion_include_path / "legion_defines.h",
                legion_spy_py=legion_bin_path / "legion_spy.py",
                legion_prof_py=legion_bin_path / "legion_prof.py",
                legion_python=legion_bin_path / "legion_python",
                legion_module=legion_module,
            )

        raise RuntimeError("Could not determine legion paths")

    if (legate_build_dir := legate_paths.legate_build_dir) is None:
        legate_build_dir = get_legate_build_dir(legate_paths.legate_dir)

    # If no local build dir found, assume legate installed into the python env
    if legate_build_dir is None:
        return installed_legion_paths(Path(sys.argv[0]).parents[1])

    # If a legate build dir was found, read `Legion_SOURCE_DIR` and
    # `Legion_BINARY_DIR` from in CMakeCache.txt, return paths into the source
    # and build dirs. This allows devs to quickly rebuild inplace and use the
    # most up-to-date versions without needing to install Legion and
    # legate_core globally.

    cmake_cache_txt = legate_build_dir / "CMakeCache.txt"

    try:
        # Test whether Legion_DIR is set. If it isn't, then we built Legion as
        # a side-effect of building legate_core
        read_cmake_cache_value(
            "Legion_DIR:PATH=Legion_DIR-NOTFOUND", cmake_cache_txt
        )
    except Exception:
        # If Legion_DIR is a valid path, check whether it's a
        # Legion build dir, i.e. `-D Legion_ROOT=/legion/build`
        legion_dir = Path(
            read_cmake_cache_value("Legion_DIR:PATH=", cmake_cache_txt)
        )
        if legion_dir.joinpath("CMakeCache.txt").exists():
            cmake_cache_txt = legion_dir / "CMakeCache.txt"

        try:
            # If Legion_SOURCE_DIR and Legion_BINARY_DIR are in CMakeCache.txt,
            # return the paths to Legion in the legate_core build dir.
            legion_source_dir = Path(
                read_cmake_cache_value(
                    "Legion_SOURCE_DIR:STATIC=", cmake_cache_txt
                )
            )
            legion_binary_dir = Path(
                read_cmake_cache_value(
                    "Legion_BINARY_DIR:STATIC=", cmake_cache_txt
                )
            )

            legion_runtime_dir = legion_binary_dir / "runtime"
            legion_bindings_dir = legion_source_dir / "bindings"

            return LegionPaths(
                legion_bin_path=legion_binary_dir / "bin",
                legion_lib_path=legion_binary_dir / "lib",
                realm_defines_h=legion_runtime_dir / "realm_defines.h",
                legion_defines_h=legion_runtime_dir / "legion_defines.h",
                legion_spy_py=legion_source_dir / "tools" / "legion_spy.py",
                legion_prof_py=legion_source_dir / "tools" / "legion_prof.py",
                legion_python=legion_binary_dir / "bin" / "legion_python",
                legion_module=legion_bindings_dir / "python" / "build" / "lib",
            )
        except Exception:
            pass

    # Otherwise return the installation paths.
    return installed_legion_paths(Path(sys.argv[0]).parents[1])
