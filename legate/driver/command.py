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
from __future__ import annotations

from typing import TYPE_CHECKING

from .ui import warn

if TYPE_CHECKING:
    from .config import Config
    from .launcher import Launcher
    from .system import System
    from .types import CommandPart

__all__ = ("CMD_PARTS",)


def cmd_bind(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    cpu_bind = config.binding.cpu_bind
    mem_bind = config.binding.mem_bind
    gpu_bind = config.binding.gpu_bind
    nic_bind = config.binding.nic_bind

    if all(x is None for x in (cpu_bind, mem_bind, gpu_bind, nic_bind)):
        return ()

    ranks = config.multi_node.ranks

    opts: CommandPart = (
        str(system.legate_paths.bind_sh_path),
        "local"
        if launcher.kind == "none" and ranks == 1
        else str(launcher.kind),
    )

    ranks_per_node = config.multi_node.ranks_per_node

    errmsg = "Number of groups in --{name}-bind not equal to --ranks-per-node"

    def check_bind_ranks(name: str, binding: str) -> None:
        if len(binding.split("/")) != ranks_per_node:
            raise RuntimeError(errmsg.format(name=name))

    bindings = (
        ("cpu", cpu_bind),
        ("gpu", gpu_bind),
        ("mem", mem_bind),
        ("nic", nic_bind),
    )
    for name, binding in bindings:
        if binding is not None:
            check_bind_ranks(name, binding)
            opts += (f"--{name}s", binding)

    return opts


def cmd_gdb(config: Config, system: System, launcher: Launcher) -> CommandPart:
    if not config.debugging.gdb:
        return ()

    if config.multi_node.ranks > 1:
        print(warn("Legate does not support gdb for multi-rank runs"))
        return ()

    return ("lldb", "--") if system.os == "Darwin" else ("gdb", "--args")


def cmd_cuda_gdb(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    if not config.debugging.cuda_gdb:
        return ()

    if config.multi_node.ranks > 1:
        print(warn("Legate does not support cuda-gdb for multi-rank runs"))
        return ()

    return ("cuda-gdb", "--args")


def cmd_nvprof(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    if not config.profiling.nvprof:
        return ()

    log_path = str(config.logging.logdir / f"legate_{launcher.rank_id}.nvvp")

    return ("nvprof", "-o", log_path)


def cmd_nsys(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    if not config.profiling.nsys:
        return ()

    log_path = str(config.logging.logdir / f"legate_{launcher.rank_id}")
    targets = config.profiling.nsys_targets
    extra = config.profiling.nsys_extra

    opts: CommandPart = ("nsys", "profile", "-t", targets, "-o", log_path)
    opts += tuple(extra)
    if "-s" not in extra:
        opts += ("-s", "none")

    return opts


def cmd_memcheck(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    memcheck = config.debugging.memcheck

    return () if not memcheck else ("compute-sanitizer",)


def cmd_nocr(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    control_replicable = not config.multi_node.not_control_replicable

    return () if control_replicable else ("--nocr",)


def cmd_module(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    module = config.other.module

    return () if module is None else ("-m", module)


def cmd_rlwrap(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    return ("rlwrap",) if config.other.rlwrap else ()


def cmd_legion(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    return (str(system.legion_paths.legion_python),)


def cmd_processor(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    # We always need one python processor per rank and no local fields
    return ("-ll:py", "1", "-lg:local", "0")


def cmd_kthreads(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    freeze_on_error = config.debugging.freeze_on_error
    gdb = config.debugging.gdb
    cuda_gdb = config.debugging.cuda_gdb

    if freeze_on_error or gdb or cuda_gdb:
        # Running with userspace threads would not allow us to inspect the
        # stacktraces of suspended threads.
        return ("-ll:force_kthreads",)

    return ()


def cmd_cpus(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    cpus = config.core.cpus

    return () if cpus == 1 else ("-ll:cpu", str(cpus))


def cmd_gpus(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    gpus = config.core.gpus

    # Make sure that we skip busy GPUs
    return () if gpus == 0 else ("-ll:gpu", str(gpus), "-cuda:skipbusy")


def cmd_openmp(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    openmp = config.core.openmp
    ompthreads = config.core.ompthreads
    numamem = config.memory.numamem

    if openmp == 0:
        return ()

    if ompthreads == 0:
        print(
            warn(
                f"Legate is ignoring request for {openmp} "
                "OpenMP processors with 0 threads"
            )
        )
        return ()

    return (
        "-ll:ocpu",
        str(openmp),
        "-ll:othr",
        str(ompthreads),
        # If we have support for numa memories then add the extra flag
        "-ll:onuma",
        f"{int(numamem > 0)}",
    )


def cmd_utility(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    utility = config.core.utility
    ranks = config.multi_node.ranks

    if utility == 1:
        return ()

    opts: CommandPart = ("-ll:util", str(utility))

    # If we are running multi-rank then make the number of active
    # message handler threads equal to our number of utility
    # processors in order to prevent head-of-line blocking
    if ranks > 1:
        opts += ("-ll:bgwork", str(utility))

    return opts


def cmd_mem(config: Config, system: System, launcher: Launcher) -> CommandPart:
    # Always specify the csize
    return ("-ll:csize", str(config.memory.sysmem))


def cmd_numamem(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    numamem = config.memory.numamem
    return () if numamem == 0 else ("-ll:nsize", str(numamem))


def cmd_fbmem(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    if config.core.gpus == 0:
        return ()

    fbmem, zcmem = config.memory.fbmem, config.memory.zcmem
    return ("-ll:fsize", str(fbmem), "-ll:zsize", str(zcmem))


def cmd_regmem(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    regmem = config.memory.regmem
    return () if regmem == 0 else ("-ll:rsize", str(regmem))


def cmd_log_levels(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    log_dir = config.logging.logdir

    levels: CommandPart = ("openmp=5",)
    opts: CommandPart = ()

    if config.profiling.profile:
        ranks = config.multi_node.ranks
        opts += ("-lg:prof", str(ranks))
        opts += ("-lg:prof_logfile", str(log_dir / "legate_%.prof"))
        levels += ("legion_prof=2",)

    # The gpu log supression may not be needed in the future.
    # Currently, the cuda hijack layer generates some spurious warnings.
    if config.core.gpus > 0:
        levels += ("gpu=5",)

    if config.debugging.dataflow or config.debugging.event:
        opts += ("-lg:spy",)
        levels += ("legion_spy=2",)

    if config.logging.user_logging_levels is not None:
        levels += (config.logging.user_logging_levels,)

    opts += ("-level", ",".join(levels))

    return opts


def cmd_log_file(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    log_dir = config.logging.logdir
    log_to_file = config.logging.log_to_file

    if log_to_file:
        return ("-logfile", str(log_dir / "legate_%.log"))

    return ()


def cmd_eager_alloc(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    eager_alloc = config.memory.eager_alloc

    return ("-lg:eager_alloc_percentage", str(eager_alloc))


def cmd_user_opts(
    config: Config, system: System, launcher: Launcher
) -> CommandPart:
    return config.user_opts


CMD_PARTS = (
    cmd_bind,
    cmd_rlwrap,
    cmd_gdb,
    cmd_cuda_gdb,
    cmd_nvprof,
    cmd_nsys,
    # Add memcheck right before the binary
    cmd_memcheck,
    # Now we're ready to build the actual command to run
    cmd_legion,
    # This has to go before script name
    cmd_nocr,
    cmd_module,
    cmd_processor,
    cmd_kthreads,
    # Translate the requests to Realm command line parameters
    cmd_cpus,
    cmd_gpus,
    cmd_openmp,
    cmd_utility,
    cmd_mem,
    cmd_numamem,
    cmd_fbmem,
    cmd_regmem,
    cmd_log_levels,
    cmd_log_file,
    cmd_eager_alloc,
    # Append user flags so they can override whatever we provided
    cmd_user_opts,
)
