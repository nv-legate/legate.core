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


class LegateRuntimeSettings(Settings):
    consensus: PrioritizedSetting[bool] = PrioritizedSetting(
        "consensus",
        "LEGATE_CONSENSUS",
        default=False,
        convert=convert_bool,
        help="""
        Whether to perform the RegionField consensus match operation on
        single-node runs (for testing). This is normally only necessary on
        multi-node runs, where all processes must collectively agree that a
        RegionField has been garbage collected at the Python level before it
        can be reused.
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
        inhibit used RegionFields from being collected and reused for new
        Stores, thus increasing memory pressure. By default this check will
        miss any RegionField cycles that the garbage collector collected during
        execution. Run gc.disable() at the beginning of the program to avoid
        this.
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
        Enable test mode. This sets alternative defaults for various other
        settings.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_gpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_gpu_chunk",
        "LEGATE_MIN_GPU_CHUNK",
        default=1048576,  # 1 << 20
        test_default=2,
        convert=convert_int,
        help="""
        If using GPUs, any task operating on arrays smaller than this will
        not be parallelized across more than one GPU.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_cpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_cpu_chunk",
        "LEGATE_MIN_CPU_CHUNK",
        default=16384,  # 1 << 14
        test_default=2,
        convert=convert_int,
        help="""
        If using CPUs, any task operating on arrays smaller than this will
        not be parallelized across more than one core.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_omp_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_omp_chunk",
        "LEGATE_MIN_OMP_CHUNK",
        default=131072,  # 1 << 17
        test_default=2,
        convert=convert_int,
        help="""
        If using OpenMP, any task operating on arrays smaller than this will
        not be parallelized across more than one OpenMP group.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    window_size: EnvOnlySetting[int] = EnvOnlySetting(
        "window_size",
        "LEGATE_WINDOW_SIZE",
        default=1,
        test_default=1,
        convert=convert_int,
        help="""
        How many Legate operations to accumulate before emitting to Legion.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    max_pending_exceptions: EnvOnlySetting[int] = EnvOnlySetting(
        "max_pending_exceptions",
        "LEGATE_MAX_PENDING_EXCEPTIONS",
        default=64,
        test_default=1,
        convert=convert_int,
        help="""
        How many possibly-exception-throwing tasks to emit before blocking.
        Legate by default does not wait for operations to complete, but instead
        "runs ahead" and continues launching work, which will complete
        asynchronously. If an operation throws an exception, then by the time
        an exception is reported execution may have progressed beyond the
        launch of the faulting operation. If you need to check for an exception
        at the exact point where it might get thrown (e.g. to catch it and
        recover gracefully), set this to 1. Note that this will introduce more
        blocking in the control logic of your program, likely reducing overall
        utilization.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    precise_exception_trace: EnvOnlySetting[bool] = EnvOnlySetting(
        "precise_exception_trace",
        "LEGATE_PRECISE_EXCEPTION_TRACE",
        default=False,
        test_default=False,
        convert=convert_bool,
        help="""
        Whether to capture the stacktrace at the point when a potentially
        faulting operation is launched, so a more accurate error location can
        be reported in case an exception is thrown.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    field_reuse_frac: EnvOnlySetting[int] = EnvOnlySetting(
        "field_reuse_frac",
        "LEGATE_FIELD_REUSE_FRAC",
        default=256,
        test_default=256,
        convert=convert_int,
        help="""
        Any allocation for more than 1/frac of available memory will count as
        multiple allocations, for purposes of triggering a consensus match.
        Only relevant for multi-node runs.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    field_reuse_freq: EnvOnlySetting[int] = EnvOnlySetting(
        "field_reuse_freq",
        "LEGATE_FIELD_REUSE_FREQ",
        default=32,
        test_default=32,
        convert=convert_int,
        help="""
        Every how many RegionField allocations to perform a consensus match
        operation. Only relevant for multi-node runs, where all processes must
        collectively agree that a RegionField has been garbage collected at the
        Python level before it can be reused.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    max_lru_length: EnvOnlySetting[int] = EnvOnlySetting(
        "max_lru_length",
        "LEGATE_MAX_LRU_LENGTH",
        default=5,
        test_default=1,
        convert=convert_int,
        help="""
        Once the last Store of a given shape is garbage collected, the
        resources associated with it are placed on an LRU queue, rather than
        getting freed immediately, in case the program creates a Store of the
        same shape in the near future. This setting controls the length of that
        LRU queue.

        This is a read-only environment variable setting used by the runtime.
        """,
    )


settings = LegateRuntimeSettings()
