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

# mypy: ignore-errors
import shutil
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager
from legion_jupyter.install_jupyter import driver, parse_args

if __name__ == "__main__":
    legate_exe = Path(shutil.which("legate"))
    legate_dir = legate_exe.parent.absolute()
    args, opts = parse_args()
    if args.json == "legion_python.json":
        # override the default one
        args.json = "legate_jupyter.json"
    args.legion_prefix = legate_dir
    kernel_file_dir = None
    kernel_name = driver(args, opts, kernel_file_dir)
    # copy the json file into ipython kernel directory
    ksm = KernelSpecManager()
    spec = ksm.get_kernel_spec(kernel_name)
    shutil.copy(args.json, spec.resource_dir)

    # TODO: copy legate_info.py and json file into legate dir
