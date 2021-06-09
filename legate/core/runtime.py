# Copyright 2021 NVIDIA Corporation
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

from legion_cffi import ffi  # Make sure we only have one ffi instance
from legion_top import cleanup_items, top_level

from .context import Context
from .corelib import CoreLib
from .legion import legate_task_postamble, legate_task_preamble, legion


class Attachment(object):
    def __init__(self, ptr, extent, region, field_id):
        self.ptr = ptr
        self.extent = extent
        self.end = ptr + extent - 1
        self.region = region
        self.field_id = field_id
        self.count = 1

    def overlaps(self, start, stop):
        if self.end < start:
            return False
        if stop < self.ptr:
            return False
        return True

    def equals(self, ptr, extent):
        return ptr == self.ptr and extent == self.extent


class Runtime(object):
    def __init__(self):
        """
        This is a class that implements the Legate runtime.
        The Runtime object provides high-level APIs for Legate libraries
        to use services in the Legion runtime. The Runtime centralizes
        resource management for all the libraries so that they can
        focus on implementing their domain logic.
        """

        self._contexts = {}
        self._context_list = []
        self._core_context = None
        self._attachments = None
        self._empty_argmap = legion.legion_argument_map_create()

        # This list maintains outstanding operations from all legate libraries
        # to be dispatched. This list allows cross library introspection for
        # Legate operations.
        self._outstanding_ops = []

        try:
            self._legion_context = top_level.context[0]
        except AttributeError:
            pass

        # Record whether we need to run finalize tasks
        # Key off whether we are being loaded in a context or not
        try:
            # Do this first to detect if we're not in the top-level task
            self._legion_context = top_level.context[0]
            self._legion_runtime = legion.legion_runtime_get_runtime()
            legate_task_preamble(self._legion_runtime, self._legion_context)
            self._finalize_tasks = True
        except AttributeError:
            self._finalize_tasks = False
            self._legion_runtime = None
            self._legion_context = None

    @property
    def legion_runtime(self):
        if self._legion_runtime is None:
            self._legion_runtime = legion.legion_runtime_get_runtime()
        return self._legion_runtime

    @property
    def legion_context(self):
        return self._legion_context

    @property
    def core_context(self):
        if self._core_context is None:
            self._core_context = self._contexts["legate.core"]
        return self._core_context

    @property
    def empty_argmap(self):
        return self._empty_argmap

    def register_library(self, library):
        libname = library.get_name()
        if libname in self._contexts:
            raise RuntimeError(
                f"library {libname} has already been registered!"
            )
        # It's important that we load the library so that its constants
        # can be used for configuration.
        self.load_library(library)
        context = Context(self, library)
        self._contexts[libname] = context
        self._context_list.append(context)
        return context

    @staticmethod
    def load_library(library):
        shared_lib_path = library.get_shared_library()
        if shared_lib_path is not None:
            header = library.get_c_header()
            if header is not None:
                ffi.cdef(header)
            shared_lib = ffi.dlopen(shared_lib_path)
            library.initialize(shared_lib)
            callback_name = library.get_registration_callback()
            callback = getattr(shared_lib, callback_name)
            callback()
        else:
            library.initialize()

    def destroy(self):
        # Destroy all libraries. Note that we should do this
        # from the lastly added one to the first one
        for context in reversed(self._context_list):
            context.destroy()
        del self._contexts
        del self._context_list
        # Clean up our attachments so that they can be collected
        self._attachments = None
        if self._finalize_tasks:
            # Run a gc and then end the legate task
            gc.collect()
            legate_task_postamble(self.legion_runtime, self.legion_context)

    def dispatch(self, op, redop=None):
        if redop:
            return op.launch(self.legion_runtime, self.legion_context, redop)
        else:
            return op.launch(self.legion_runtime, self.legion_context)

    def add_attachment(self, ptr, extent, region, field_id):
        key = (ptr, extent)
        if self._attachments is None:
            self._attachments = dict()
        elif key in self._attachments:
            # If we find exactly the match then we know by definition that
            # nobody overlaps with this attachment or it wouldn't exist
            self._attachments[key].count += 1
            return
        # Otherwise iterate over attachments and look for aliases which are bad
        end = ptr + extent - 1
        for attachment in self._attachments.values():
            if attachment.overlaps(ptr, end):
                assert not attachment.equals(ptr, extent)
                raise RuntimeError(
                    "Illegal aliased attachments not supported by Legate"
                )
        self._attachments[key] = Attachment(ptr, extent, region, field_id)

    def find_attachment(self, ptr, extent):
        if self._attachments is None:
            return None
        key = (ptr, extent)
        if key in self._attachments:
            attachment = self._attachments[key]
            assert attachment.count > 0
            return (attachment.region, attachment.field_id)
        # Otherwise look for aliases which are bad
        end = ptr + extent - 1
        for attachment in self._attachments.values():
            if attachment.overlaps(ptr, end):
                assert not attachment.equals(ptr, extent)
                raise RuntimeError(
                    "Illegal aliased attachments not supported by Legate"
                )
        return None

    def remove_attachment(self, ptr, extent):
        key = (ptr, extent)
        if key not in self._attachments:
            raise RuntimeError("Unable to find attachment to remove")
        attachment = self._attachments[key]
        assert attachment.count > 0
        if attachment.count == 1:
            del self._attachments[key]
        else:
            attachment.count -= 1


_runtime = Runtime()
_runtime.register_library(CoreLib())


def _cleanup_legate_runtime():
    global _runtime
    _runtime.destroy()
    del _runtime
    gc.collect()


cleanup_items.append(_cleanup_legate_runtime)


def get_legion_runtime():
    return _runtime.legion_runtime


def get_legion_context():
    return _runtime.legion_context


def legate_add_library(library):
    _runtime.register_library(library)


# These functions provide support for deduplicating attaches across libraries
def legate_add_attachment(ptr, extent, region, field_id):
    _runtime.add_attachment(ptr, extent, region, field_id)


def legate_find_attachment(ptr, extent):
    return _runtime.find_attachment(ptr, extent)


def legate_remove_attachment(ptr, extent):
    _runtime.remove_attachment(ptr, extent)


def get_legate_runtime():
    return _runtime
