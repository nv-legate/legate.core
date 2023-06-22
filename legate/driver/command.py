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

import argparse
from typing import TYPE_CHECKING

from .. import install_info
from ..util.ui import warn

if TYPE_CHECKING:
    from ..util.system import System
    from ..util.types import CommandPart
    from .config import ConfigProtocol
    from .launcher import Launcher

__all__ = ("CMD_PARTS_LEGION", "CMD_PARTS_CANONICAL")


# this will be replaced by bind.sh with the actual computed rank at runtime
LEGATE_GLOBAL_RANK_SUBSTITUTION = "%%LEGATE_GLOBAL_RANK%%"


def cmd_bind(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    ranks = config.multi_node.ranks

    if ranks > 1 and len(install_info.networks) == 0:
        raise RuntimeError(
            "multi-rank run was requested, but Legate was not built with "
            "networking support"
        )

    if launcher.kind == "none":
        bind_launcher_arg = "local" if ranks == 1 else "auto"
    else:
        bind_launcher_arg = launcher.kind

    opts: CommandPart = (
        str(system.legate_paths.bind_sh_path),
        "--launcher",
        bind_launcher_arg,
    )

    ranks_per_node = config.multi_node.ranks_per_node

    errmsg = "Number of groups in --{name}-bind not equal to --ranks-per-node"

    def check_bind_ranks(name: str, binding: str) -> None:
        if len(binding.split("/")) != ranks_per_node:
            raise RuntimeError(errmsg.format(name=name))

    bindings = (
        ("cpu", config.binding.cpu_bind),
        ("gpu", config.binding.gpu_bind),
        ("mem", config.binding.mem_bind),
        ("nic", config.binding.nic_bind),
    )
    for name, binding in bindings:
        if binding is not None:
            check_bind_ranks(name, binding)
            opts += (f"--{name}s", binding)

    if config.info.bind_detail:
        opts += ("--debug",)

    return opts + ("--",)


def cmd_gdb(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    if not config.debugging.gdb:
        return ()

    if config.multi_node.ranks > 1:
        print(warn("Legate does not support gdb for multi-rank runs"))
        return ()

    return ("lldb", "--") if system.os == "Darwin" else ("gdb", "--args")


def cmd_cuda_gdb(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    if not config.debugging.cuda_gdb:
        return ()

    if config.multi_node.ranks > 1:
        print(warn("Legate does not support cuda-gdb for multi-rank runs"))
        return ()

    return ("cuda-gdb", "--args")


def cmd_nvprof(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    if not config.profiling.nvprof:
        return ()

    log_path = str(
        config.logging.logdir
        / f"legate_{LEGATE_GLOBAL_RANK_SUBSTITUTION}.nvvp"
    )

    return ("nvprof", "-o", log_path)


def cmd_nsys(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    if not config.profiling.nsys:
        return ()

    log_path = str(
        config.logging.logdir / f"legate_{LEGATE_GLOBAL_RANK_SUBSTITUTION}"
    )
    targets = config.profiling.nsys_targets
    extra = config.profiling.nsys_extra

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample")
    parser.add_argument("-t", "--targets")
    nsys_parsed_args, unparsed = parser.parse_known_args(extra)

    if nsys_parsed_args.targets:
        raise RuntimeError(
            "please pass targets as arguments to --nsys"
            "rather than using --nsys-extra"
        )

    opts: CommandPart = ("nsys", "profile", "-t", targets, "-o", log_path)
    opts += tuple(extra)
    if not nsys_parsed_args.sample:
        opts += ("-s", "none")

    return opts


def cmd_valgrind(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    valgrind = config.debugging.valgrind

    return () if not valgrind else ("valgrind",)


def cmd_memcheck(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    memcheck = config.debugging.memcheck

    return () if not memcheck else ("compute-sanitizer",)


def cmd_nocr(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    control_replicable = not config.multi_node.not_control_replicable

    return () if control_replicable else ("--nocr",)


def cmd_module(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    module = config.other.module
    cprofile = config.profiling.cprofile

    if cprofile and module is not None:
        raise ValueError("Only one of --module or --cprofile may be used")

    if module is not None:
        return ("-m", module)

    if cprofile:
        log_path = str(
            config.logging.logdir
            / f"legate_{LEGATE_GLOBAL_RANK_SUBSTITUTION}.cprof"
        )
        return ("-m", "cProfile", "-o", log_path)

    return ()


def cmd_rlwrap(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    return ("rlwrap",) if config.other.rlwrap else ()


def cmd_legion(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    return (str(system.legion_paths.legion_python),)


def cmd_python_processor(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    # We always need one python processor per rank
    return ("-ll:py", "1")


def cmd_kthreads(
    config: ConfigProtocol, system: System, launcher: Launcher
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
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    cpus = config.core.cpus

    return () if cpus == 1 else ("-ll:cpu", str(cpus))


def cmd_gpus(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    gpus = config.core.gpus

    if gpus > 0 and not install_info.use_cuda:
        raise RuntimeError(
            "--gpus was requested, but this build does not have CUDA "
            "support enabled"
        )

    # Make sure that we skip busy GPUs
    return () if gpus == 0 else ("-ll:gpu", str(gpus), "-cuda:skipbusy")


def cmd_openmp(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    openmp = config.core.openmp
    ompthreads = config.core.ompthreads
    numamem = config.memory.numamem

    if openmp > 0 and not install_info.use_openmp:
        raise RuntimeError(
            "--omps was requested, but this build does not have OpenMP "
            "support enabled"
        )

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


def cmd_bgwork(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    ranks = config.multi_node.ranks
    utility = config.core.utility

    opts: CommandPart = ()

    # If we are running multi-rank then make the number of active
    # message handler threads equal to our number of utility
    # processors in order to prevent head-of-line blocking
    if ranks > 1:
        opts += ("-ll:bgwork", str(max(utility, 2)))

    if ranks > 1 and "ucx" in install_info.networks:
        opts += ("-ll:bgworkpin", "1")

    return opts


def cmd_utility(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    utility = config.core.utility

    if utility == 1:
        return ()

    return ("-ll:util", str(utility))


def cmd_mem(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    # Always specify the csize
    return ("-ll:csize", str(config.memory.sysmem))


def cmd_numamem(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    numamem = config.memory.numamem
    return () if numamem == 0 else ("-ll:nsize", str(numamem))


def cmd_fbmem(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    if config.core.gpus == 0:
        return ()

    fbmem, zcmem = config.memory.fbmem, config.memory.zcmem
    return ("-ll:fsize", str(fbmem), "-ll:zsize", str(zcmem))


def cmd_regmem(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    regmem = config.memory.regmem
    return () if regmem == 0 else ("-ll:rsize", str(regmem))


def cmd_network(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    # Don't initialize a Realm network module if running on a single rank
    return () if config.multi_node.ranks > 1 else ("-ll:networks", "none")


def cmd_log_levels(
    config: ConfigProtocol, system: System, launcher: Launcher
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

    if (
        config.debugging.dataflow
        or config.debugging.event
        or config.debugging.collective
    ):
        opts += ("-lg:spy",)
        levels += ("legion_spy=2",)

    if config.logging.user_logging_levels is not None:
        levels += (config.logging.user_logging_levels,)

    opts += ("-level", ",".join(levels))

    return opts


def cmd_log_file(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    log_dir = config.logging.logdir
    log_to_file = config.logging.log_to_file

    if log_to_file:
        return ("-logfile", str(log_dir / "legate_%.log"), "-errlevel", "4")

    return ()


def cmd_eager_alloc(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    eager_alloc = config.memory.eager_alloc

    return ("-lg:eager_alloc_percentage", str(eager_alloc))


def cmd_user_script(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    return () if config.user_script is None else (config.user_script,)


def cmd_user_opts(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    return config.user_opts


def cmd_python(
    config: ConfigProtocol, system: System, launcher: Launcher
) -> CommandPart:
    return ("python",)


_CMD_PARTS_SHARED = (
    # This has to go before script name
    cmd_nocr,
    cmd_kthreads,
    # Translate the requests to Realm command line parameters
    cmd_cpus,
    cmd_gpus,
    cmd_openmp,
    cmd_utility,
    cmd_bgwork,
    cmd_mem,
    cmd_numamem,
    cmd_fbmem,
    cmd_regmem,
    cmd_network,
    cmd_log_levels,
    cmd_log_file,
    cmd_eager_alloc,
)

CMD_PARTS_LEGION = (
    (
        cmd_bind,
        cmd_rlwrap,
        cmd_gdb,
        cmd_cuda_gdb,
        cmd_nvprof,
        cmd_nsys,
        # Add memcheck right before the binary
        cmd_memcheck,
        # Add valgrind right before the binary
        cmd_valgrind,
        # Now we're ready to build the actual command to run
        cmd_legion,
        # This has to go before script name
        cmd_python_processor,
        cmd_module,
    )
    + _CMD_PARTS_SHARED
    + (
        # User script
        cmd_user_script,
        # Append user flags so they can override whatever we provided
        cmd_user_opts,
    )
)

CMD_PARTS_CANONICAL = (
    (
        # Executable name that will get stripped by the runtime
        cmd_python,
        # User script
        cmd_user_script,
    )
    + _CMD_PARTS_SHARED
    + (
        # Append user flags so they can override whatever we provided
        cmd_user_opts,
    )
)
