#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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
from typing import Literal, Protocol, Tuple

# --- Types -------------------------------------------------------------------

Req = str
Reqs = Tuple[Req, ...]
OSType = Literal["linux", "osx"]


def V(version: str) -> tuple[int, ...]:
    padded_version = (version.split(".") + ["0"])[:2]
    return tuple(int(x) for x in padded_version)


class SectionConfig(Protocol):
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
    ctk_version: str
    compilers: bool
    os: OSType

    header = "cuda"

    @property
    def conda(self) -> Reqs:
        if self.ctk_version == "none":
            return ()

        deps = (
            f"cuda-version={self.ctk_version}",  # runtime
            "cutensor>=1.3.3",  # runtime
            "nccl",  # runtime
            "pynvml",  # tests
        )

        # gcc 11.3 is incompatible with nvcc <= 11.5.
        if (
            self.compilers
            and self.os == "linux"
            and (V(self.ctk_version) <= V("11.5"))
        ):
            deps += (
                "gcc_linux-64<=11.2",
                "gxx_linux-64<=11.2",
            )

        return deps

    def __str__(self) -> str:
        if self.ctk_version == "none":
            return ""

        return f"-cuda{self.ctk_version}"


@dataclass(frozen=True)
class BuildConfig(SectionConfig):
    compilers: bool = True
    openmpi: bool = True
    ucx: bool = True

    header = "build"

    @property
    def conda(self) -> Reqs:
        pkgs = (
            # 3.25.0 triggers gitlab.kitware.com/cmake/cmake/-/issues/24119
            "cmake>=3.24,!=3.25.0",
            "cython",
            "git",
            "make",
            "rust",
            "ninja",
            "scikit-build>=0.13.1",
            "setuptools>=60",
            "zlib",
            "numba",
            "valgrind",
        )
        if self.compilers:
            pkgs += ("c-compiler", "cxx-compiler")
        if self.openmpi:
            pkgs += ("openmpi",)
        if self.ucx:
            pkgs += ("ucx>=1.14",)
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
        )

    @property
    def pip(self) -> Reqs:
        return ("tifffile",)


@dataclass(frozen=True)
class DocsConfig(SectionConfig):
    header = "docs"

    @property
    def conda(self) -> Reqs:
        return ("pandoc", "doxygen")

    @property
    def pip(self) -> Reqs:
        return (
            "ipython",
            "jinja2",
            "markdown<3.4.0",
            "pydata-sphinx-theme>=0.13",
            "myst-parser",
            "nbsphinx",
            "sphinx-copybutton",
            "sphinx>=4.4.0",
        )


@dataclass(frozen=True)
class EnvConfig:
    use: str
    python: str
    os: OSType
    ctk: str
    compilers: bool
    openmpi: bool
    ucx: bool

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
        return CUDAConfig(self.ctk, self.compilers, self.os)

    @property
    def build(self) -> BuildConfig:
        return BuildConfig(self.compilers, self.openmpi, self.ucx)

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
        return f"environment-{self.use}-{self.os}-py{self.python}{self.cuda}{self.build}.yaml"  # noqa


# --- Setup -------------------------------------------------------------------

PYTHON_VERSIONS = ("3.9", "3.10", "3.11")

CTK_VERSIONS = (
    "none",
    "10.2",
    "11.0",
    "11.1",
    "11.2",
    "11.3",
    "11.4",
    "11.5",
    "11.6",
    "11.7",
    "11.8",
    "12.0",
    # TODO: libcublas 12.1 not available on conda-forge as of 2023-06-12
    # "12.1",
)

OS_NAMES: Tuple[OSType, ...] = ("linux", "osx")


ENV_TEMPLATE = """\
name: legate-{use}
channels:
  - conda-forge
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

ALL_CONFIGS = [
    EnvConfig("test", python, "linux", ctk, compilers, openmpi, ucx)
    for python in PYTHON_VERSIONS
    for ctk in CTK_VERSIONS
    for compilers in (True, False)
    for openmpi in (True, False)
    for ucx in (True, False)
] + [
    EnvConfig("test", python, "osx", "none", compilers, openmpi, False)
    for python in PYTHON_VERSIONS
    for compilers in (True, False)
    for openmpi in (True, False)
]

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
        default=None,
        help="Python version to generate for, (default: all python versions)",
    )
    parser.add_argument(
        "--ctk",
        choices=CTK_VERSIONS,
        default=None,
        dest="ctk_version",
        help="CTK version to generate for (default: all CTK versions)",
    )
    parser.add_argument(
        "--os",
        choices=OS_NAMES,
        default=None,
        help="OS to generate for (default: all OSes)",
    )
    parser.add_argument(
        "--compilers",
        action=BooleanFlag,
        dest="compilers",
        default=None,
        help="Whether to include conda compilers or not (default: both)",
    )
    parser.add_argument(
        "--openmpi",
        action=BooleanFlag,
        dest="openmpi",
        default=None,
        help="Whether to include openmpi or not (default: both)",
    )
    parser.add_argument(
        "--ucx",
        action=BooleanFlag,
        dest="ucx",
        default=None,
        help="Whether to include UCX or not (default: both)",
    )

    args = parser.parse_args(sys.argv[1:])

    configs = ALL_CONFIGS

    if args.python is not None:
        configs = (x for x in configs if x.python == args.python)
    if args.ctk_version is not None:
        configs = (
            x for x in configs if x.cuda.ctk_version == args.ctk_version
        )
    if args.compilers is not None:
        configs = (x for x in configs if x.build.compilers == args.compilers)
    if args.os is not None:
        configs = (x for x in configs if x.os == args.os)
    if args.openmpi is not None:
        configs = (x for x in configs if x.build.openmpi == args.openmpi)
    if args.ucx is not None:
        configs = (x for x in configs if x.build.ucx == args.ucx)

    for config in configs:
        conda_sections = indent(
            "".join(s.format("conda") for s in config.sections if s.conda),
            "  ",
        )

        pip_sections = indent(
            "".join(s.format("pip") for s in config.sections if s.pip), "    "
        )

        print(f"--- generating: {config.filename}")
        out = ENV_TEMPLATE.format(
            use=config.use,
            python=config.python,
            conda_sections=conda_sections,
            pip=PIP_TEMPLATE.format(pip_sections=pip_sections),
        )
        with open(f"{config.filename}", "w") as f:
            f.write(out)
