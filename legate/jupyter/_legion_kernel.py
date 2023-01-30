#!/usr/bin/env python

# Copyright 2022 Los Alamos National Laboratory, NVIDIA Corporation
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

import sys
from contextlib import contextmanager
from typing import Any, Iterator, TextIO

from ipykernel.ipkernel import IPythonKernel  # type: ignore [import]

__version__ = "0.1"


@contextmanager
def reset_stdout(stdout: TextIO) -> Iterator[None]:
    _stdout = sys.stdout
    sys.stdout = stdout
    yield
    sys.stdout = _stdout


class LegionKernel(IPythonKernel):  # type: ignore [misc,no-any-unimported]
    implementation = "legion_kernel"
    implementation_version = __version__
    banner = "Legion IPython Kernel for SM"
    language = "python"
    language_version = __version__
    language_info = {
        "name": "legion_kernel",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }

    def __init__(self, **kwargs: Any) -> None:
        with reset_stdout(open("/dev/stdout", "w")):
            print("Initializing Legion kernel for single- or multi-node.")
        super().__init__(**kwargs)


if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp  # type: ignore [import]

    IPKernelApp.launch_instance(kernel_class=LegionKernel)
