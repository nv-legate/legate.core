#!/usr/bin/env python3

# Copyright 2023 NVIDIA Corporation
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

import re
import sys
from argparse import ArgumentParser, FileType
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, unique
from typing import TextIO

_HELP_MSG = """
Process the log from a Legate testing run into a json file that can be
visualized using Chrome's built-in trace viewer. Different processes in the
visualization correspond to different shards, and different threads to
different tests. Only the first GPU-enabled test run in the source file will be
processed. Assumes the testing run was performed with --debug.
"""

_INVOCATION_PAT = r"^\+.*legate .* --gpu-bind ([0-9,]+) ([^ ]+\.py)"

_TIMESTAMP_PAT = r"[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9]"

_TEST_RESULT_PAT = (
    r"^\[(PASS|FAIL)\] \(GPU\) ([0-9]+\.[0-9]+)s {("
    + _TIMESTAMP_PAT
    + r"), "
    + _TIMESTAMP_PAT
    + r"} ([^ ]+\.py)"
)

_FULL_RESULT_PAT = r"Results .* ([0-9]+) files passed"


@dataclass
class Test:
    tid: int
    file: str
    result: str
    ts: int
    dur: int
    gpu: int

    def to_json(self) -> str:
        return (
            f'{{"name": "{self.result}"'
            f', "ph": "X"'
            f', "pid": {self.gpu}'
            f', "tid": {self.tid}'
            f', "ts": {self.ts}'
            f', "dur": {self.dur}'
            f', "args": {{"file": "{self.file}"}}'
            "},\n"
        )


@unique
class State(Enum):
    BEFORE_GPU_STAGE = 1
    IN_GPU_STAGE = 2
    AFTER_GPU_STAGE = 3
    DONE = 4


class LineParser:
    def __init__(self) -> None:
        self._state = State.BEFORE_GPU_STAGE
        self._gpu_for_test: dict[str, int] = {}
        self._next_tid = 0
        self._tests: list[Test] = []

    def _find_gpu(self, file: str) -> int:
        # We have to do it like this because the invocation and result lines
        # report the filename differently.
        gpu = None
        for k, v in self._gpu_for_test.items():
            if k.endswith(file):
                gpu = v
                break
        assert gpu is not None
        return gpu

    def parse(self, line: str) -> None:
        if self._state == State.BEFORE_GPU_STAGE:
            if "Entering stage: GPU" in line:
                self._state = State.IN_GPU_STAGE
            return
        if self._state == State.AFTER_GPU_STAGE:
            if (m := re.search(_FULL_RESULT_PAT, line)) is not None:
                num_tests = int(m.group(1))
                assert num_tests == len(self._tests)
                self._state = State.DONE
            return
        if self._state == State.DONE:
            return

        assert self._state == State.IN_GPU_STAGE
        if "Exiting stage" in line:
            self._state = State.AFTER_GPU_STAGE
            return
        if (m := re.search(_INVOCATION_PAT, line)) is not None:
            gpu = int(m.group(1).split(",")[0])  # just keep the first GPU
            file = m.group(2)
            self._gpu_for_test[file] = gpu
            return
        if (m := re.search(_TEST_RESULT_PAT, line)) is not None:
            result = m.group(1)
            dur = int(float(m.group(2)) * 1000000)
            start = datetime.strptime(m.group(3), "%H:%M:%S.%f")
            ts = (start - datetime(1900, 1, 1)) // timedelta(microseconds=1)
            file = m.group(4)
            gpu = self._find_gpu(file)
            self._tests.append(
                Test(self._next_tid, file, result, ts, dur, gpu)
            )
            self._next_tid += 1

    def write_json(self, out: TextIO) -> None:
        # First sort tests by start time, then duration.
        self._tests.sort(key=lambda test: (test.ts, test.dur))
        for i, test in enumerate(self._tests):
            test.tid = i
        out.write("[\n")
        for test in self._tests:
            out.write(test.to_json())
        out.write("]\n")


if __name__ == "__main__":
    arg_parser = ArgumentParser(description=_HELP_MSG)
    arg_parser.add_argument(
        "input",
        type=FileType("r"),
        help="Input filename",
    )
    arg_parser.add_argument(
        "output",
        nargs="?",
        type=FileType("w"),
        default=sys.stdout,
        help="Output filename; if not given print to stdout",
    )
    args = arg_parser.parse_args()

    line_parser = LineParser()
    for line in args.input:
        line_parser.parse(line)
    line_parser.write_json(args.output)
