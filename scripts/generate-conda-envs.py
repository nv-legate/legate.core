#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See the LICENSE file for details.
#
from __future__ import annotations

from argparse import Action, ArgumentParser
from dataclasses import dataclass
from textwrap import indent
from typing import Literal, Tuple, Union

# --- Types -------------------------------------------------------------------

Req = str
Reqs = Tuple[Req, ...]
OSType = Literal["linux", "osx"]


def V(version: str) -> tuple[int, ...]:
    padded_version = (version.split(".") + ["0", "0"])[:3]
    return tuple(int(x) for x in padded_version)


def drop_patch(version: str) -> str:
    return ".".join(version.split(".")[:2])


class SectionConfig:
    header: str

    @property
    def conda(self) -> Reqs:
        return ()

    @property
    def pip(self) -> Reqs:
        return ()

    def __str__(self) -> str:
        return self.header

    def format(self, kind: str) -> str:
        return SECTION_TEMPLATE.format(
            header=self.header,
            reqs="- "
            + "\n- ".join(self.conda if kind == "conda" else self.pip),
        )


@dataclass(frozen=True)
class CUDAConfig(SectionConfig):
    ctk_version: Union[str, None]
    compilers: bool
    os: OSType

    header = "cuda"

    @property
    def conda(self) -> Reqs:
        if not self.ctk_version:
            return ()

        deps = (
            f"cuda-version={drop_patch(self.ctk_version)}",  # runtime
            # cuTensor package notes:
            # - We are pinning to 1.X major version.
            #   See https://github.com/nv-legate/cunumeric/issues/1092.
            # - The cuTensor packages on the nvidia channel are not well
            #   structured. The multiple levels of packages are not connected
            #   by strict dependencies, and the CTK compatibility is encoded
            #   in the package name, rather than a constraint or label.
            #   For now we pin to the conda-forge versions (which use build
            #   numbers starting with h).
            "cutensor=1.7*=h*",  # runtime
            "nccl",  # runtime
            "pynvml",  # tests
        )

        if V(self.ctk_version) < (12, 0, 0):
            deps += (f"cudatoolkit={self.ctk_version}",)
        else:
            deps += (
                "cuda-cccl",  # no cuda-cccl-dev package on the nvidia channel
                "cuda-cudart-dev",
                "cuda-cudart-static",
                "cuda-driver-dev",
                "cuda-nvml-dev",
                "cuda-nvtx",  # no cuda-nvtx-dev package on the nvidia channel
                "libcublas-dev",
                "libcufft-dev",
                "libcurand-dev",
                "libcusolver-dev",
                "libcusparse-dev",
                "libnvjitlink-dev",
            )

        if self.compilers:
            if self.os == "linux":
                if V(self.ctk_version) < (12, 0, 0):
                    deps += (f"nvcc_linux-64={drop_patch(self.ctk_version)}",)
                else:
                    deps += ("cuda-nvcc",)

                # gcc 11.3 is incompatible with nvcc < 11.6.
                if V(self.ctk_version) < (11, 6, 0):
                    deps += (
                        "gcc_linux-64<=11.2",
                        "gxx_linux-64<=11.2",
                    )
                else:
                    deps += (
                        "gcc_linux-64=11.*",
                        "gxx_linux-64=11.*",
                    )

        return deps

    def __str__(self) -> str:
        if not self.ctk_version:
            return ""

        return f"-cuda{self.ctk_version}"


@dataclass(frozen=True)
class BuildConfig(SectionConfig):
    compilers: bool
    openmpi: bool
    ucx: bool
    os: OSType

    header = "build"

    @property
    def conda(self) -> Reqs:
        pkgs = (
            # 3.25.0 triggers gitlab.kitware.com/cmake/cmake/-/issues/24119
            "cmake>=3.24,!=3.25.0",
            "cython>=3",
            "git",
            "make",
            "rust",
            "ninja",
            "openssl",
            "pkg-config",
            "scikit-build>=0.13.1",
            "setuptools>=60",
            "zlib",
            "numba",
        )
        if self.compilers:
            pkgs += ("c-compiler", "cxx-compiler")
        if self.openmpi:
            # Using a more recent version of OpenMPI in combination with the
            # system compilers fails with: "Could NOT find MPI (missing:
            # MPI_CXX_FOUND CXX)". The reason is that conda-forge's libmpi.so
            # v5 is linked against the conda-forge libstdc++.so. CMake will not
            # use the mpicc-suggested host compiler (conda-forge's gcc) when
            # running the FindMPI tests. Instead it will use the "main"
            # compiler it was configured with, i.e. the system compiler, which
            # links agains the system libstdc++.so, causing libmpi.so's symbol
            # version checks to fail. Using v5+ OpenMPI from conda-forge
            # requires using the conda-forge compilers.
            pkgs += ("openmpi<5",)
        if self.ucx:
            pkgs += ("ucx>=1.14",)
        if self.os == "linux":
            pkgs += ("elfutils",)
        return sorted(pkgs)

    def __str__(self) -> str:
        val = "-compilers" if self.compilers else ""
        val += "-openmpi" if self.openmpi else ""
        val += "-ucx" if self.ucx else ""
        return val


@dataclass(frozen=True)
class RuntimeConfig(SectionConfig):
    header = "runtime"

    @property
    def conda(self) -> Reqs:
        return (
            "cffi",
            "llvm-openmp",
            "numpy>=1.22",
            "libblas=*=*openblas*",
            "openblas=*=*openmp*",
            # work around https://github.com/StanfordLegion/legion/issues/1500
            "openblas<=0.3.21",
            "opt_einsum",
            "scipy",
            "typing_extensions",
        )


@dataclass(frozen=True)
class TestsConfig(SectionConfig):
    header = "tests"

    @property
    def conda(self) -> Reqs:
        return (
            "clang-tools>=8",
            "clang>=8",
            "colorama",
            "coverage",
            "mock",
            "mypy>=0.961",
            "pre-commit",
            "pytest-cov",
            "pytest-lazy-fixture",
            "pytest-mock",
            "pytest",
            "types-docutils",
            "pynvml",
            "tifffile",
        )

    @property
    def pip(self) -> Reqs:
        return ()


@dataclass(frozen=True)
class DocsConfig(SectionConfig):
    header = "docs"

    @property
    def conda(self) -> Reqs:
        return (
            "pandoc",
            "doxygen",
            "ipython",
            "jinja2",
            "markdown<3.4.0",
            "pydata-sphinx-theme>=0.13",
            "myst-parser",
            "nbsphinx",
            "sphinx-copybutton",
            "sphinx>=4.4.0",
        )

    @property
    def pip(self) -> Reqs:
        return ()


@dataclass(frozen=True)
class EnvConfig:
    use: str
    python: str
    os: OSType
    ctk_version: Union[str, None]
    compilers: bool
    openmpi: bool
    ucx: bool

    @property
    def channels(self) -> str:
        channels = []
        if self.ctk_version and V(self.ctk_version) >= (12, 0, 0):
            channels.append(f"nvidia/label/cuda-{self.ctk_version}")
        channels.append("conda-forge")
        return "- " + "\n- ".join(channels)

    @property
    def sections(self) -> Tuple[SectionConfig, ...]:
        return (
            self.cuda,
            self.build,
            self.runtime,
            self.tests,
            self.docs,
        )

    @property
    def cuda(self) -> CUDAConfig:
        return CUDAConfig(self.ctk_version, self.compilers, self.os)

    @property
    def build(self) -> BuildConfig:
        return BuildConfig(self.compilers, self.openmpi, self.ucx, self.os)

    @property
    def runtime(self) -> RuntimeConfig:
        return RuntimeConfig()

    @property
    def tests(self) -> TestsConfig:
        return TestsConfig()

    @property
    def docs(self) -> DocsConfig:
        return DocsConfig()

    @property
    def filename(self) -> str:
        return f"environment-{self.use}-{self.os}-py{self.python}{self.cuda}{self.build}"  # noqa


# --- Setup -------------------------------------------------------------------

PYTHON_VERSIONS = ("3.9", "3.10", "3.11")

OS_NAMES: Tuple[OSType, ...] = ("linux", "osx")


ENV_TEMPLATE = """\
name: legate-{use}
channels:
{channels}
dependencies:

  - python={python},!=3.9.7  # avoid https://bugs.python.org/issue45121

{conda_sections}{pip}
"""

SECTION_TEMPLATE = """\
# {header}
{reqs}

"""

PIP_TEMPLATE = """\
  - pip
  - pip:
{pip_sections}
"""


# --- Code --------------------------------------------------------------------


class BooleanFlag(Action):
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


if __name__ == "__main__":
    import sys

    parser = ArgumentParser()
    parser.add_argument(
        "--python",
        choices=PYTHON_VERSIONS,
        default="3.10",
        help="Python version to generate for",
    )
    parser.add_argument(
        "--ctk",
        dest="ctk_version",
        help="CTK version to generate for",
    )
    parser.add_argument(
        "--os",
        choices=OS_NAMES,
        default=("osx" if sys.platform == "darwin" else "linux"),
        help="OS to generate for",
    )
    parser.add_argument(
        "--compilers",
        action=BooleanFlag,
        dest="compilers",
        default=False,
        help="Whether to include conda compilers or not",
    )
    parser.add_argument(
        "--openmpi",
        action=BooleanFlag,
        dest="openmpi",
        default=False,
        help="Whether to include openmpi or not",
    )
    parser.add_argument(
        "--ucx",
        action=BooleanFlag,
        dest="ucx",
        default=False,
        help="Whether to include UCX or not",
    )

    parser.add_argument(
        "--sections",
        nargs="*",
        help="""List of sections exclusively selected for inclusion in the
        generated environment file.""",
    )

    args = parser.parse_args(sys.argv[1:])

    if (
        args.ctk_version
        and V(args.ctk_version) >= (12, 0, 0)
        and len(args.ctk_version.split(".")) != 3
    ):
        # This is necessary to match on the exact label on the nvidia channel
        raise ValueError("CTK 12 versions must be in the form 12.X.Y")

    selected_sections = None

    if args.sections is not None:
        selected_sections = set(args.sections)

    def section_selected(section):
        if not selected_sections:
            return True

        if selected_sections and str(section) in selected_sections:
            return True

        return False

    config = EnvConfig(
        "test",
        args.python,
        args.os,
        args.ctk_version,
        args.compilers,
        args.openmpi,
        args.ucx,
    )

    conda_sections = indent(
        "".join(
            s.format("conda")
            for s in config.sections
            if s.conda and section_selected(s)
        ),
        "  ",
    )

    pip_sections = indent(
        "".join(
            s.format("pip")
            for s in config.sections
            if s.pip and section_selected(s)
        ),
        "    ",
    )

    filename = config.filename
    if args.sections:
        filename = config.filename + "-partial"

    print(f"--- generating: {filename}.yaml")
    out = ENV_TEMPLATE.format(
        use=config.use,
        channels=config.channels,
        python=config.python,
        conda_sections=conda_sections,
        pip=PIP_TEMPLATE.format(pip_sections=pip_sections)
        if pip_sections
        else "",
    )
    with open(f"{filename}.yaml", "w") as f:
        f.write(out)
