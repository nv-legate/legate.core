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

from dataclasses import dataclass
from typing import Literal, Protocol

from jinja2 import Template
from typing_extensions import TypeAlias

# --- Types -------------------------------------------------------------------

Req: TypeAlias = str
Reqs: TypeAlias = tuple[Req, ...]
OSType: TypeAlias = Literal["linux", "darwin"]


class SectionConfig(Protocol):
    heading: str

    @property
    def conda(self) -> Reqs:
        return ()

    @property
    def pip(self) -> Reqs:
        return ()

    def __str__(self) -> str:
        return self.heading


@dataclass(frozen=True)
class CUDAConfig(SectionConfig):
    ctk_version: str | None

    header = "cuda"

    @property
    def conda(self) -> Reqs:
        if self.ctk_version is None:
            return ()

        return (
            f"cudatoolkit=={self.ctk_version}",  # runtime
            "cutensor",  # runtime
            "nccl",  # runtime
            "pynvml",  # tests
        )

    def __str__(self) -> str:
        if self.ctk_version == "none":
            return ""

        return f"-cuda-{self.ctk_version}"


@dataclass(frozen=True)
class BuildConfig(SectionConfig):
    compilers: bool = True
    openmpi: bool = True

    header = "build"

    @property
    def conda(self) -> Reqs:
        pkgs = (
            "cmake>=3.24",
            "git",
            "make",
            "ninja",
            "scikit-build>=0.13.1",
            "setuptools>=60",
            "zlib",
        )
        if self.compilers:
            pkgs += ("c-compiler", "cxx-compiler")
        if self.openmpi:
            pkgs += ("openmpi",)
        return sorted(pkgs)

    def __str__(self) -> str:
        val = "-compilers" if self.compilers else "-no-compilers"
        val += "-with-openmpi" if self.openmpi else "-without-openmpi"
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
            "opt_einsum",
            "pyarrow>=5",
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
    def pip(self) -> Reqs:
        return (
            "jinja2",
            "markdown<3.4.0",
            "pydata-sphinx-theme",
            "recommonmark",
            "sphinx-copybutton",
            "sphinx-markdown-tables",
            "sphinx>=4.4.0",
        )


@dataclass(frozen=True)
class EnvConfig:
    use: str
    python: str
    os: OSType
    ctk: str | None
    compilers: bool
    openmpi: bool

    @property
    def sections(self) -> tuple[SectionConfig, ...]:
        return (
            self.cuda,
            self.build,
            self.runtime,
            self.tests,
            self.docs,
        )

    @property
    def cuda(self) -> CUDAConfig:
        return CUDAConfig(self.ctk)

    @property
    def build(self) -> BuildConfig:
        return BuildConfig(self.compilers, self.openmpi)

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
        python = f"py{self.python.replace('.', '')}"
        return f"environment-{self.use}-{self.os}-{python}{self.cuda}{self.build}.yaml"  # noqa


# --- Setup -------------------------------------------------------------------

PYTHON_VERSIONS = ("3.8", "3.9", "3.10")

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
)

OS_NAMES: tuple[OSType, ...] = ("linux", "osx")


ENV_TEMPLATE = Template(
    """
name: legate-core-test
channels:
  - conda-forge
dependencies:

  - python={{ python }}
  {% if conda_sections %}
  {% for section in conda_sections %}

  # {{ section.header }}
  {% for req in section.conda %}
  - {{ req }}
  {% endfor %}
  {% endfor %}
  {% endif %}
  {% if pip_sections %}

  - pip
  - pip:
      {% for section in pip_sections %}

      # {{ section.header }}
      {% for req in section.pip %}
      - {{ req }}
      {% endfor %}
      {% endfor %}
  {% endif %}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

ALL_CONFIGS = [
    EnvConfig("test", python, "linux", ctk, compilers, openmpi)
    for python in PYTHON_VERSIONS
    for ctk in CTK_VERSIONS
    for compilers in (True, False)
    for openmpi in (True, False)
] + [
    EnvConfig("test", python, "darwin", "none", compilers, openmpi)
    for python in PYTHON_VERSIONS
    for compilers in (True, False)
    for openmpi in (True, False)
]

# --- Code --------------------------------------------------------------------

if __name__ == "__main__":

    import sys
    from argparse import ArgumentParser, BooleanOptionalAction

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
        action=BooleanOptionalAction,
        default=None,
        help="Whether to include conda compilers or not (default: both)",
    )
    parser.add_argument(
        "--openmpi",
        action=BooleanOptionalAction,
        default=None,
        help="Whether to include openmpi or not (default: both)",
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

    for config in configs:
        conda_sections = [
            section for section in config.sections if section.conda
        ]
        pip_sections = [section for section in config.sections if section.pip]

        print(f"--- generating: {config.filename}")
        out = ENV_TEMPLATE.render(
            python=config.python,
            conda_sections=conda_sections,
            pip_sections=pip_sections,
        )
        with open(f"{config.filename}", "w") as f:
            f.write(out)
