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

    @property
    def filename_component(self) -> str:
        if self.ctk_version is None:
            return ""

        return f"-cuda-{self.ctk_version}"


class BuildConfig(SectionConfig):
    header = "build"

    @property
    def conda(self) -> Reqs:
        return (
            "c-compiler",
            "cmake>=3.24",
            "cxx-compiler",
            "gcc_linux-64 # [linux64]",
            "git",
            "make",
            "ninja",
            "openmpi",
            "scikit-build>=0.13.1",
            "setuptools>=60",
            "sysroot_linux-64==2.17 # [linux64]",
            "zlib",
        )


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

    @property
    def sections(self) -> tuple[SectionConfig, ...]:
        return (
            self.cuda,
            BuildConfig(),
            RuntimeConfig(),
            TestsConfig(),
            DocsConfig(),
        )

    @property
    def cuda(self) -> CUDAConfig:
        return CUDAConfig(self.ctk)

    @property
    def filename(self) -> str:
        python = f"py{self.python.replace('.', '')}"
        cuda = self.cuda.filename_component
        return f"environment-{self.use}-{self.os}-{python}{cuda}.yaml"


# --- Setup -------------------------------------------------------------------

PYTHON_VERSIONS = ("3.8", "3.9", "3.10")

CTK_VERSIONS = (
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

CONFIGS = [
    EnvConfig("test", python, "linux", ctk)
    for python in PYTHON_VERSIONS
    for ctk in CTK_VERSIONS + (None,)
] + [EnvConfig("test", python, "darwin", None) for python in PYTHON_VERSIONS]

# --- Code --------------------------------------------------------------------

for config in CONFIGS:
    conda_sections = [section for section in config.sections if section.conda]
    pip_sections = [section for section in config.sections if section.pip]

    print(f"------- {config.filename}")
    out = ENV_TEMPLATE.render(
        python=config.python,
        conda_sections=conda_sections,
        pip_sections=pip_sections,
    )
    with open(f"conda/{config.filename}", "w") as f:
        f.write(out)
