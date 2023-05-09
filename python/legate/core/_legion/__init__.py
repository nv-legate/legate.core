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

from .env import LEGATE_MAX_DIM, LEGATE_MAX_FIELDS
from .field import FieldID
from .future import Future, FutureMap
from .geometry import Point, Rect, Domain
from .operation import (
    Acquire,
    Attach,
    Copy,
    Detach,
    Dispatchable,
    Fill,
    IndexAttach,
    IndexCopy,
    IndexDetach,
    IndexFill,
    Release,
    InlineMapping,
)
from .partition import IndexPartition, Partition
from .partition_functor import (
    PartitionFunctor,
    PartitionByRestriction,
    PartitionByImage,
    PartitionByImageRange,
    EqualPartition,
    PartitionByWeights,
    PartitionByDomain,
)
from .region import Region, OutputRegion, PhysicalRegion
from .space import IndexSpace, FieldSpace
from .task import ArgumentMap, Fence, Task, IndexTask
from .transform import Transform, AffineTransform
from .util import (
    dispatch,
    BufferBuilder,
    ExternalResources,
    FieldListLike,
    legate_task_preamble,
    legate_task_postamble,
    legate_task_progress,
)

__all__ = (
    "Acquire",
    "AffineTransform",
    "ArgumentMap",
    "Attach",
    "BufferBuilder",
    "Copy",
    "Detach",
    "dispatch",
    "Domain",
    "EqualPartition",
    "ExternalResources",
    "Fence",
    "FieldID",
    "FieldListLike",
    "FieldSpace",
    "Fill",
    "Future",
    "FutureMap",
    "IndexAttach",
    "IndexCopy",
    "IndexDetach",
    "IndexFill",
    "IndexPartition",
    "IndexSpace",
    "IndexTask",
    "InlineMapping",
    "OutputRegion",
    "Partition",
    "PartitionByDomain",
    "PartitionByImage",
    "PartitionByImageRange",
    "PartitionByRestriction",
    "PartitionByWeights",
    "PartitionFunctor",
    "PhysicalRegion",
    "Point",
    "Rect",
    "Region",
    "Release",
    "Task",
    "Transform",
    "legate_task_preamble",
    "legate_task_postamble",
    "legate_task_progress",
    "LEGATE_MAX_DIM",
    "LEGATE_MAX_FIELDS",
)
