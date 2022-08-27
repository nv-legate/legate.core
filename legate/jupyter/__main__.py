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
import os
import shutil
import sys

from jupyter_client.kernelspec import KernelSpecManager


def find_python_module(legate_dir: str) -> str:
    lib_dir = os.path.join(legate_dir, "lib")
    python_lib = None
    for f in os.listdir(lib_dir):
        if f.startswith("python") and not os.path.isfile(f):
            if python_lib is not None:
                print(
                    "WARNING: Found multiple python supports in your Legion "
                    "installation."
                )
                print("Using the following one: " + str(python_lib))
            else:
                python_lib = os.path.join(
                    lib_dir, os.path.join(f, "site-packages")
                )

    if python_lib is None:
        raise Exception("Cannot find a Legate python library")
    return python_lib


if __name__ == "__main__":
    try:
        legate_dir = os.environ["LEGATE_DIR"]
        legate_dir = os.path.abspath(legate_dir)
    except KeyError:
        print(
            "Please specify the legate installation dir "
            "by setting LEGATE_DIR"
        )
        sys.exit(1)

    python_lib_dir = find_python_module(legate_dir)
    sys.path.append(python_lib_dir)
    from install_jupyter import driver, parse_args

    args, opts = parse_args()
    if args.json == "legion_python.json":
        # override the default one
        args.json = "legate_jupyter.json"
    args.legion_prefix = legate_dir + "/bin"
    kernel_file_dir = python_lib_dir
    kernel_name = driver(args, opts, kernel_file_dir)
    # copy the json file into ipython kernel directory
    ksm = KernelSpecManager()
    spec = ksm.get_kernel_spec(kernel_name)
    shutil.copy(args.json, spec.resource_dir)

    # TODO: copy legate_info.py and json file into legate dir
