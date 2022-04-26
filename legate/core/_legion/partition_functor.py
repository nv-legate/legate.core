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

from typing import TYPE_CHECKING, Any, Union

from .. import ffi, legion
from .future import FutureMap
from .geometry import Point

if TYPE_CHECKING:
    from . import FieldID, IndexPartition, IndexSpace, Rect, Region, Transform


class PartitionFunctor:
    """
    PartitionFunctor objects provide a common interface to computing
    IndexPartition objects using Legion's support for dependent partitioning.
    Each kind of dependent partitioning operator in Legion can be accessed
    through a custom PartitionFunctor.
    """

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: int,
        part_id: int,
    ) -> Any:
        """
        The generic interface for computing an IndexPartition

        Parameters
        ----------
        runtime : legion_runtime_t
            Handle for the Legion runtime from get_legion_runtime()
        context : legion_context_t
            The Legion context from get_legion_context()
        parent : IndexSpace
            The parent index space for which we're computing this partition
        color_space : IndexSpace
            The color space defining the names of subspaces to be created
        kind : legion_partition_kind_t
            Description of the disjointness and completeness property
            (if they are known) for the partition
        part_id : int
            Desired name for the color of this partition in the parent
            index space's color space (if any)
        """
        raise NotImplementedError("implement in derived classes")


class PartitionByRestriction(PartitionFunctor):
    def __init__(self, transform: Transform, extent: Rect) -> None:
        """
        PartitionByRestriction constructs a tesselated IndexPartition where an
        IndexSpace is created for each Point in the color space by projecting
        the point through the transform to create the lower bound point and
        then adding the extent. Note that this can be used to create both
        disjoint and aliased partitions, depending on the properties of the
        transform and the extent.
        """

        self.transform = transform
        self.extent = extent

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        return legion.legion_index_partition_create_by_restriction(
            runtime,
            context,
            parent.handle,
            color_space.handle,
            self.transform.raw(),
            self.extent.raw(),
            kind,
            part_id,
        )


class PartitionByImage(PartitionFunctor):
    def __init__(
        self,
        region: Region,
        part: IndexPartition,
        field: Union[int, FieldID],
        mapper: int = 0,
        tag: int = 0,
    ) -> None:
        """
        PartitionByImage projects an existing IndexPartition through a field
        of Points that point from one LogicalRegion into an IndexSpace.
        """

        self.region = region
        self.part = part
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        return legion.legion_index_partition_create_by_image(
            runtime,
            context,
            parent.handle,
            self.part.handle,
            self.region.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class PartitionByImageRange(PartitionFunctor):
    def __init__(
        self,
        region: Region,
        part: IndexPartition,
        field: Union[int, FieldID],
        mapper: int = 0,
        tag: int = 0,
    ) -> None:
        """
        PartitionByImageRange projects an existing IndexPartition through a
        field of Rects that point from one LogicalRegion into a range of points
        in a destination IndexSpace.
        """

        self.region = region
        self.part = part
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        return legion.legion_index_partition_create_by_image_range(
            runtime,
            context,
            parent.handle,
            self.part.handle,
            self.region.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class PartitionByPreimage(PartitionFunctor):
    def __init__(
        self,
        projection: IndexPartition,
        region: Region,
        parent: Region,
        field: Union[int, FieldID],
        mapper: int = 0,
        tag: int = 0,
    ) -> None:
        """
        Partition by preimage induces a partition on the index space of
        'region' by taking a field of region that points into the index
        partition 'projection" and reversing the mapping.
        """
        self.projection = projection
        self.region = region
        self.parent = parent
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        return legion.legion_index_partition_create_by_preimage(
            runtime,
            context,
            self.projection.handle,
            self.region.handle,
            self.parent.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class PartitionByPreimageRange(PartitionFunctor):
    def __init__(
        self,
        projection: IndexPartition,
        region: Region,
        parent: Region,
        field: Union[int, FieldID],
        mapper: int = 0,
        tag: int = 0,
    ) -> None:
        """
        Partition by preimage range induces a partition on the index space of
        'region' by taking a field of region that points into ranges of the
        index partition 'projection" and reversing the mapping.
        """
        self.projection = projection
        self.region = region
        self.parent = parent
        self.field = field
        self.mapper = mapper
        self.tag = tag

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        return legion.legion_index_partition_create_by_preimage_range(
            runtime,
            context,
            self.projection.handle,
            self.region.handle,
            self.parent.handle,
            self.field.fid if isinstance(self.field, FieldID) else self.field,
            color_space.handle,
            kind,
            part_id,
            self.mapper,
            self.tag,
        )


class EqualPartition(PartitionFunctor):
    """
    EqualPartition will construct an IndexPartition that creates IndexSpace
    children with roughly equal numbers of points in each child.
    """

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        return legion.legion_index_partition_create_equal(
            runtime,
            context,
            parent.handle,
            color_space.handle,
            1,
            part_id,
        )


class PartitionByWeights(PartitionFunctor):
    def __init__(self, weights: FutureMap) -> None:
        """
        PartitionByWeights will construct an IndexPartition with the number of
        points in each child IndexSpace being allocated proportionally to the
        the relative weights.
        """

        self.weights = weights

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: int,
        part_id: int,
    ) -> Any:
        if isinstance(self.weights, FutureMap):
            return legion.legion_index_partition_create_by_weights_future_map(
                runtime,
                context,
                parent.handle,
                self.weights.handle,
                color_space.handle,
                1,
                part_id,
            )
        else:
            raise TypeError("Unsupported type for PartitionByWeights")


class PartitionByDomain(PartitionFunctor):
    def __init__(self, domains: Union[FutureMap, dict[Point, Rect]]) -> None:
        """
        PartitionByDomain will construct an IndexPartition given an explicit
        mapping of colors to domains.

        Parameters
        ----------
        domains : FutureMap | dict[Point, Rect]
        """
        self.domains = domains

    def partition(
        self,
        runtime: legion.legion_runtime_t,
        context: legion.legion_context_t,
        parent: IndexSpace,
        color_space: IndexSpace,
        kind: Any,
        part_id: int,
    ) -> Any:
        if isinstance(self.domains, FutureMap):
            return legion.legion_index_partition_create_by_domain_future_map(
                runtime,
                context,
                parent.handle,
                self.domains.handle,
                color_space.handle,
                True,  # perform_intersections
                kind,
                part_id,
            )
        elif isinstance(self.domains, dict):
            num_domains = len(self.domains)
            assert num_domains <= color_space.get_volume()
            colors = ffi.new("legion_domain_point_t[%d]" % num_domains)
            domains = ffi.new("legion_domain_t[%d]" % num_domains)
            for (i, (point, rect)) in enumerate(self.domains.items()):
                colors[i] = point.raw()
                domains[i] = rect.raw()
            return legion.legion_index_partition_create_by_domain(
                runtime,
                context,
                parent.handle,
                colors,
                domains,
                num_domains,
                color_space.handle,
                True,  # perform_intersections
                kind,
                part_id,
            )
        else:
            raise TypeError("Unsupported type for PartitionByDomain")


# TODO more kinds of partition functors here
