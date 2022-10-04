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

import gc
import inspect
import sys
from collections import deque
from types import FunctionType
from typing import Any, Set, Union


def _skip(src: Any, dst: Any) -> bool:
    return (
        src is sys.modules
        or isinstance(src, FunctionType)
        and hasattr(src, "__globals__")
        and src.__globals__ is dst
    )


def _find_cycles(root: Any, all_ids: Set[int]) -> bool:
    opened: dict[int, int] = {}
    closed: Set[int] = set()
    stack = [root]
    while len(stack) > 0:
        dst = stack[-1]
        if id(dst) in opened:
            if opened[id(dst)] == len(stack):
                del opened[id(dst)]
                closed.add(id(dst))
            else:
                print("found cycle!")
                print("  tail:")
                _bfs(dst, root, all_ids)
                print("  cycle:")
                _bfs(dst, dst, all_ids)
                return True
            stack.pop()
        elif id(dst) in closed:
            stack.pop()
        else:
            opened[id(dst)] = len(stack)
            for src in gc.get_referrers(dst):
                if id(src) in all_ids and not _skip(src, dst):
                    stack.append(src)
    return False


def _find_field(src: Any, dst: Any) -> Union[str, None]:
    if type(src) == dict:
        for k, v in src.items():
            if v is dst and isinstance(k, str):
                return f'["{k}"]'
    if type(src) == tuple:
        for k, v in enumerate(src):
            if v is dst:
                return f"[{k}]"
    if type(src) == list:
        for i, v in enumerate(src):
            if v is dst:
                return f"[{i}]"
    try:
        for fld in dir(src):
            try:
                if hasattr(src, fld) and getattr(src, fld) is dst:
                    return "." + fld
            except Exception:
                pass
    except Exception:
        pass
    try:
        for fld in vars(src):
            try:
                if hasattr(src, fld) and getattr(src, fld) is dst:
                    return "." + fld
            except Exception:
                pass
    except Exception:
        pass
    try:
        for fld, val in inspect.getmembers(src):
            if val is dst:
                return "." + fld
    except Exception:
        pass
    return None


def _bfs(begin: Any, end: Any, all_ids: Set[int]) -> None:
    parent = {}
    q = deque([begin])
    while len(q) > 0:
        src = q.popleft()
        for dst in gc.get_referents(src):
            if id(dst) not in all_ids or id(dst) in parent or _skip(src, dst):
                continue
            parent[id(dst)] = src
            if dst is end:
                print(f"    {hex(id(dst))}: {type(dst)}")
                while True:
                    src = parent[id(dst)]
                    fld = _find_field(src, dst)
                    if fld is None:
                        print("     ^")
                    else:
                        print(f"     ^ {fld}")
                    print(f"    {hex(id(src))}: {type(src)}")
                    dst = src
                    if dst is begin:
                        break
                return
            q.append(dst)
    print(f"    {hex(id(end))}: {type(end)}")
    print("     ^")
    print("    ???")
    print("     ^")
    print(f"    {hex(id(begin))}: {type(begin)}")


def find_cycles() -> bool:
    from .store import RegionField

    found_cycles = False
    all_objs = gc.get_objects()
    all_ids = set(id(obj) for obj in all_objs)
    for obj in all_objs:
        if isinstance(obj, RegionField):
            print(
                f"looking for cycles involving {hex(id(obj))}, "
                f"of type {type(obj)}"
            )
            if _find_cycles(obj, all_ids):
                found_cycles = True
    return found_cycles
