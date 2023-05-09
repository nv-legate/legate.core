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

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from ._legion import Future


class PendingException:
    def __init__(
        self,
        exn_types: list[type],
        future: Future,
        tb_repr: Optional[str] = None,
    ):
        self._exn_types = exn_types
        self._future = future
        self._tb_repr = tb_repr

    def raise_exception(self) -> None:
        buf = self._future.get_buffer()
        (raised,) = struct.unpack("?", buf[:1])
        if not raised:
            return
        (exn_index, error_size) = struct.unpack("iI", buf[1:9])
        error_message = buf[9 : 9 + error_size].decode()
        exn_type = self._exn_types[exn_index]
        exn_reraised = exn_type(error_message)
        if self._tb_repr is not None:
            error_message += "\n" + self._tb_repr[:-1]  # remove extra newline
        exn_original = exn_type(error_message)
        raise exn_reraised from exn_original
