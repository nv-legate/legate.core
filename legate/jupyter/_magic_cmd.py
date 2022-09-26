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
import json
import os
import sys

from IPython.core.magic import Magics, line_magic, magics_class
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

cmd_dict = {
    "cpus": "Number of CPUs to use per rank",
    "gpus": "Number of GPUs to use per rank",
    "omps": "Number of OpenMP groups to use per rank",
    "ompthreads": "Number of threads per OpenMP group",
    "utility": "Number of Utility processors per rank",
    "sysmem": "Amount of DRAM memory per rank (in MBs)",
    "numamem": "Amount of DRAM memory per NUMA domain per rank (in MBs)",
    "fbmem": "Amount of framebuffer memory per GPU (in MBs)",
    "zcmem": "Amount of zero-copy memory per rank (in MBs)",
    "regmem": "Amount of registered CPU-side pinned memory per rank (in MBs)",
    "nodes": "Number of nodes to use",
}


class LegateInfo(object):
    def __init__(self, filename: str) -> None:
        self.config_dict = dict()
        # check if the json file is in the ipython kernel directory
        try:
            ksm = KernelSpecManager()
            spec = ksm.get_kernel_spec("legate_kernel_nocr")
        except NoSuchKernel:
            print(
                "Can not find the json file in the "
                "IPython kernel directory, please "
                "make sure the kernel has been installed."
            )
            sys.exit(1)
        filename_with_path = os.path.join(spec.resource_dir, filename)
        with open(filename_with_path) as json_file:
            json_dict = json.load(json_file)
            if missing := (set(cmd_dict) - set(json_dict)):
                raise RuntimeError(f"Expected keys {missing!r} are missing")
            for key in cmd_dict.keys():
                self.config_dict[key] = json_dict[key]["value"]

    def __repr__(self) -> str:
        out_str = ""
        for key, value in self.config_dict.items():
            out_str += f"{cmd_dict[key]}: {value}\n"
        return out_str[:-1]


@magics_class
class LegateInfoMagics(Magics):
    __slots__ = ["legate_json"]

    def __init__(self, shell):
        super(LegateInfoMagics, self).__init__(shell)
        self.legate_json = LegateInfo("legate_jupyter.json")

    @line_magic
    def legate_info(self, line: str) -> None:
        print(self.legate_json)
