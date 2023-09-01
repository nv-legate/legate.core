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

from setuptools import find_packages
from skbuild import setup

import versioneer

setup(
    name="legate-core",
    version=versioneer.get_version(),
    description="legate.core - The Foundation for All Legate Libraries",
    url="https://github.com/nv-legate/legate.core",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    extras_require={
        "test": [
            "colorama",
            "coverage",
            "mock",
            "mypy>=0.961",
            "pynvml",
            "pytest-cov",
            "pytest",
        ]
    },
    packages=find_packages(
        where=".",
        include=[
            "legate",
            "legate.*",
            "legate.core",
            "legate.core.*",
            "legate.timing",
            "legate.timing.*",
        ],
    ),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "legate = legate.driver:main",
            "legate-jupyter = legate.jupyter:main",
            "lgpatch = legate.lgpatch:main",
        ],
    },
    scripts=["bind.sh"],
    cmdclass=versioneer.get_cmdclass(),
    install_requires=[
        "cffi",
        "numpy>=1.22",
        "typing_extensions",
    ],
    zip_safe=False,
)
