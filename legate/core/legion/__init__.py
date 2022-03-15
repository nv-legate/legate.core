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

import os

from .geometry import Point, Rect, Domain
from .operation import (
    Acquire,
    Attach,
    Copy,
    Detach,
    Fill,
    IndexAttach,
    IndexCopy,
    IndexDetach,
    IndexFill,
    Release,
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
from .task import ArgumentMap, Fence, Task, IndexTask, InlineMapping
from .transform import Transform, AffineTransform
from .util import (
    _pending_unordered,
    _pending_deletions,
    dispatch,
    BufferBuilder,
    ExternalResources,
    FieldID,
    Future,
    FutureMap,
    FieldListLike,
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
)

assert "LEGATE_MAX_DIM" in os.environ
LEGATE_MAX_DIM = int(os.environ["LEGATE_MAX_DIM"])

assert "LEGATE_MAX_FIELDS" in os.environ
LEGATE_MAX_FIELDS = int(os.environ["LEGATE_MAX_FIELDS"])
