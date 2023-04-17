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
from __future__ import annotations

from .util.settings import (
    EnvOnlySetting,
    PrioritizedSetting,
    Settings,
    convert_bool,
    convert_int,
)

__all__ = ("settings",)


def _extract_define(name: str) -> str:
    from .install_info import env_defaults_header as header

    lines = (line for line in header.split("\n") if line.startswith("#define"))
    for line in lines:
        tokens = line.split(" ")
        if tokens[1].strip() == name:
            return tokens[2].strip()
    raise RuntimeError()


class LegateRuntimeSettings(Settings):
    consensus: PrioritizedSetting[bool] = PrioritizedSetting(
        "consensus",
        "LEGATE_CONSENSUS",
        default=False,
        convert=convert_bool,
        help="""
        Whether to enable consensus match on single node (for testing).
        """,
    )

    cycle_check: PrioritizedSetting[bool] = PrioritizedSetting(
        "cycle_check",
        "LEGATE_CYCLE_CHECK",
        default=False,
        convert=convert_bool,
        help="""
        Whether to check for reference cycles involving RegionField objects on
        exit (developer option). When such cycles arise during execution they
        stop used RegionFields from being collected and reused for new Stores,
        thus increasing memory pressure. By default this check will miss any
        RegionField cycles the garbage collector collected during execution.

        Run gc.disable() at the beginning of the program to avoid this.
        """,
    )

    future_leak_check: PrioritizedSetting[bool] = PrioritizedSetting(
        "future_leak_check",
        "LEGATE_FUTURE_LEAK_CHECK",
        default=False,
        convert=convert_bool,
        help="""
        Whether to check for reference cycles keeping Future/FutureMap objects
        alive after Legate runtime exit (developer option). Such leaks can
        result in Legion runtime shutdown hangs.
        """,
    )

    test: EnvOnlySetting[bool] = EnvOnlySetting(
        "test",
        "LEGATE_TEST",
        default=False,
        convert=convert_bool,
        help="""
        Whether to enable test execution.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_gpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_gpu_chunk",
        "LEGATE_MIN_GPU_CHUNK",
        default=int(_extract_define("MIN_GPU_CHUNK_DEFAULT")),
        test_default=int(_extract_define("MIN_GPU_CHUNK_TEST")),
        convert=convert_int,
        help="""
        Minimum chunk size to enable GPU execution.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_cpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_cpu_chunk",
        "LEGATE_MIN_CPU_CHUNK",
        default=int(_extract_define("MIN_CPU_CHUNK_DEFAULT")),
        test_default=int(_extract_define("MIN_CPU_CHUNK_TEST")),
        convert=convert_int,
        help="""
        Minimum chunk size to enable CPU execution.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_omp_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_omp_chunk",
        "LEGATE_MIN_OMP_CHUNK",
        default=int(_extract_define("MIN_OMP_CHUNK_DEFAULT")),
        test_default=int(_extract_define("MIN_OMP_CHUNK_TEST")),
        convert=convert_int,
        help="""
        Minimum chunk size to enable CPU execution.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    window_size: EnvOnlySetting[int] = EnvOnlySetting(
        "window_size",
        "LEGATE_WINDOW_SIZE",
        default=int(_extract_define("WINDOW_SIZE_DEFAULT")),
        test_default=int(_extract_define("WINDOW_SIZE_TEST")),
        convert=convert_int,
        help="""
        Window size.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    max_pending_exceptions: EnvOnlySetting[int] = EnvOnlySetting(
        "max_pending_exceptions",
        "LEGATE_MAX_PENDING_EXCEPTIONS",
        default=int(_extract_define("MAX_PENDING_EXCEPTIONS_DEFAULT")),
        test_default=int(_extract_define("MAX_PENDING_EXCEPTIONS_TEST")),
        convert=convert_int,
        help="""
        Maximum number of pending exceptions.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    precise_exception_trace: EnvOnlySetting[bool] = EnvOnlySetting(
        "precise_exception_trace",
        "LEGATE_PRECISE_EXCEPTION_TRACE",
        default=bool(_extract_define("PRECISE_EXCEPTION_TRACE_DEFAULT")),
        test_default=bool(_extract_define("PRECISE_EXCEPTION_TRACE_TEST")),
        convert=convert_bool,
        help="""
        Whether to enable precise exception traces.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    field_reuse_frac: EnvOnlySetting[int] = EnvOnlySetting(
        "field_reuse_frac",
        "LEGATE_FIELD_REUSE_FRAC",
        default=int(_extract_define("FIELD_REUSE_FRAC_DEFAULT")),
        test_default=int(_extract_define("FIELD_REUSE_FRAC_TEST")),
        convert=convert_int,
        help="""
        Field re-use fraction.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    field_reuse_freq: EnvOnlySetting[int] = EnvOnlySetting(
        "field_reuse_freq",
        "LEGATE_FIELD_REUSE_FREQ",
        default=int(_extract_define("FIELD_REUSE_FREQ_DEFAULT")),
        test_default=int(_extract_define("FIELD_REUSE_FREQ_TEST")),
        convert=convert_int,
        help="""
        Field re-use frequency.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    max_lru_length: EnvOnlySetting[int] = EnvOnlySetting(
        "max_lru_length",
        "LEGATE_MAX_LRU_LENGTH",
        default=int(_extract_define("MAX_LRU_LENGTH_DEFAULT")),
        test_default=int(_extract_define("MAX_LRU_LENGTH_TEST")),
        convert=convert_int,
        help="""
        Maximum LRU cache size.

        This is a read-only environment variable setting used by the runtime.
        """,
    )


settings = LegateRuntimeSettings()
