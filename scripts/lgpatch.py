#! /usr/bin/env legate
# Copyright 2022 NVIDIA Corporation
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
import sys
import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter

KNOWN_PATCHES = {"numpy": "cunumeric"}

newline = "\n"

DESCRIPTION = textwrap.dedent(
    f"""
Patch existing libraries with legate equivalents.

Currently the following patching can be applied:

{newline.join(f'    {key} -> {value}' for key, value in KNOWN_PATCHES.items())}

"""
)

EPILOG = """
Any additional command line arguments are passed on to PROG as-is
"""


def parse_args():
    parser = ArgumentParser(
        prog="lgpatch",
        description=DESCRIPTION,
        allow_abbrev=False,
        add_help=False,
        epilog=EPILOG,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prog", metavar="PROG", help="The legate program to run"
    )
    parser.add_argument(
        "-patch",
        action="store",
        nargs="+",
        help="Patch the specified libraries",
        default=[],
    )

    # legate parser intercepts "-h" and "--help" earlier
    if "-patch:help" in sys.argv:
        parser.print_help()
        sys.exit()

    return parser.parse_known_args()


def do_patch(name: str) -> None:
    if name not in KNOWN_PATCHES:
        raise ValueError(f"No patch available for module {name}")
    if name in sys.modules:
        raise RuntimeError(f"Cannot patch {name} -- it is already loaded")

    cuname = KNOWN_PATCHES[name]
    try:
        module = __import__(cuname)
        sys.modules[name] = module
    except ImportError:
        raise RuntimeError(f"Could not import patch module {cuname}")


if __name__ == "__main__":
    args, extra = parse_args()

    for name in args.patch:
        do_patch(name)

    sys.argv[:] = [args.prog] + extra

    with open(args.prog) as f:
        exec(f.read(), {"__name__": "__main__"})
