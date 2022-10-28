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

from typing import Any

class legion_runtime_t: ...
class legion_context_t: ...
class legion_phase_barrier_t: ...

LEGION_DISJOINT_COMPLETE_KIND: int
LEGION_DISJOINT_INCOMPLETE_KIND: int

LEGION_EXTERNAL_INSTANCE: int

LEGION_COMPUTE_KIND: int
SYSTEM_MEM: int

LEGION_REDOP_BASE: int
LEGION_TYPE_TOTAL: int

LEGION_READ_ONLY: int
LEGION_READ_WRITE: int
LEGION_EXCLUSIVE: int
LEGION_NO_ACCESS: int
LEGION_WRITE_DISCARD: int

LEGION_REDOP_KIND_SUM: int
LEGION_REDOP_KIND_DIFF: int
LEGION_REDOP_KIND_PROD: int
LEGION_REDOP_KIND_DIV: int
LEGION_REDOP_KIND_MAX: int
LEGION_REDOP_KIND_MIN: int
LEGION_REDOP_KIND_OR: int
LEGION_REDOP_KIND_AND: int
LEGION_REDOP_KIND_XOR: int

LEGION_TYPE_BOOL: int
LEGION_TYPE_INT8: int
LEGION_TYPE_INT16: int
LEGION_TYPE_INT32: int
LEGION_TYPE_INT64: int
LEGION_TYPE_UINT8: int
LEGION_TYPE_UINT16: int
LEGION_TYPE_UINT32: int
LEGION_TYPE_UINT64: int
LEGION_TYPE_FLOAT16: int
LEGION_TYPE_FLOAT32: int
LEGION_TYPE_FLOAT64: int
LEGION_TYPE_COMPLEX64: int
LEGION_TYPE_COMPLEX128: int

LEGATE_CORE_TUNABLE_WINDOW_SIZE: int
LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE: int
LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY: int
LEGATE_CORE_TUNABLE_MAX_LRU_LENGTH: int
LEGATE_CORE_TUNABLE_MAX_PENDING_EXCEPTIONS: int
LEGATE_CORE_TUNABLE_PRECISE_EXCEPTION_TRACE: int
LEGATE_CORE_TUNABLE_TOTAL_CPUS: int
LEGATE_CORE_TUNABLE_TOTAL_OMPS: int
LEGATE_CORE_TUNABLE_TOTAL_GPUS: int
LEGATE_CORE_TUNABLE_NUM_PIECES: int
LEGATE_CORE_TUNABLE_NUM_NODES: int
LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME: int
LEGATE_CORE_TUNABLE_NCCL_NEEDS_BARRIER: int

MAX_TYPE_NUMBER: int

def legion_acquire_launcher_add_field(*args: Any) -> Any: ...
def legion_acquire_launcher_create(*args: Any) -> Any: ...
def legion_acquire_launcher_destroy(*args: Any) -> Any: ...
def legion_acquire_launcher_execute(*args: Any) -> Any: ...
def legion_argument_map_create(*args: Any) -> Any: ...
def legion_argument_map_destroy(*args: Any) -> Any: ...
def legion_argument_map_from_future_map(*args: Any) -> Any: ...
def legion_argument_map_set_future(*args: Any) -> Any: ...
def legion_argument_map_set_point(*args: Any) -> Any: ...
def legion_attach_external_resources(*args: Any) -> Any: ...
def legion_attach_launcher_add_cpu_soa_field(*args: Any) -> Any: ...
def legion_attach_launcher_create(*args: Any) -> Any: ...
def legion_attach_launcher_destroy(*args: Any) -> Any: ...
def legion_attach_launcher_execute(*args: Any) -> Any: ...
def legion_attach_launcher_set_mapped(*args: Any) -> Any: ...
def legion_attach_launcher_set_restricted(*args: Any) -> Any: ...
def legion_attach_launcher_set_provenance(*args: Any) -> Any: ...
def legion_auto_generate_id(*args: Any) -> Any: ...
def legion_context_consensus_match(*args: Any) -> Any: ...
def legion_context_progress_unordered_operations(*args: Any) -> Any: ...
def legion_copy_launcher_add_dst_field(*args: Any) -> Any: ...
def legion_copy_launcher_add_dst_indirect_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_add_dst_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_add_dst_region_requirement_logical_region_reduction(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_add_src_field(*args: Any) -> Any: ...
def legion_copy_launcher_add_src_indirect_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_add_src_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_create(*args: Any) -> Any: ...
def legion_copy_launcher_destroy(*args: Any) -> Any: ...
def legion_copy_launcher_execute(*args: Any) -> Any: ...
def legion_copy_launcher_set_point(*args: Any) -> Any: ...
def legion_copy_launcher_set_possible_dst_indirect_out_of_range(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_set_possible_src_indirect_out_of_range(
    *args: Any,
) -> Any: ...
def legion_copy_launcher_set_sharding_space(*args: Any) -> Any: ...
def legion_copy_launcher_set_provenance(*args: Any) -> Any: ...
def legion_copy_launcher_set_mapper_arg(*args: Any) -> Any: ...
def legion_detach_external_resources(*args: Any) -> Any: ...
def legion_domain_affine_transform_identity(*args: Any) -> Any: ...
def legion_domain_empty(*args: Any) -> Any: ...
def legion_domain_get_volume(*args: Any) -> Any: ...
def legion_domain_is_dense(*args: Any) -> Any: ...
def legion_domain_point_origin(*args: Any) -> Any: ...
def legion_domain_transform_identity(*args: Any) -> Any: ...
def legion_external_resources_destroy(*args: Any) -> Any: ...
def legion_field_allocator_allocate_field(*args: Any) -> Any: ...
def legion_field_allocator_allocate_field_future(*args: Any) -> Any: ...
def legion_field_allocator_create(*args: Any) -> Any: ...
def legion_field_allocator_destroy(*args: Any) -> Any: ...
def legion_field_allocator_free_field(*args: Any) -> Any: ...
def legion_field_allocator_free_field_unordered(*args: Any) -> Any: ...
def legion_field_space_create(*args: Any) -> Any: ...
def legion_field_space_destroy_unordered(*args: Any) -> Any: ...
def legion_fill_launcher_create_from_future(*args: Any) -> Any: ...
def legion_fill_launcher_destroy(*args: Any) -> Any: ...
def legion_fill_launcher_execute(*args: Any) -> Any: ...
def legion_fill_launcher_set_point(*args: Any) -> Any: ...
def legion_fill_launcher_set_sharding_space(*args: Any) -> Any: ...
def legion_fill_launcher_set_provenance(*args: Any) -> Any: ...
def legion_future_destroy(*args: Any) -> Any: ...
def legion_future_from_untyped_pointer(*args: Any) -> Any: ...
def legion_future_get_untyped_pointer(*args: Any) -> Any: ...
def legion_future_get_untyped_size(*args: Any) -> Any: ...
def legion_future_get_void_result(*args: Any) -> Any: ...
def legion_future_is_ready_subscribe(*args: Any) -> Any: ...
def legion_future_map_construct_from_futures(*args: Any) -> Any: ...
def legion_future_map_destroy(*args: Any) -> Any: ...
def legion_future_map_get_future(*args: Any) -> Any: ...
def legion_future_map_reduce(*args: Any) -> Any: ...
def legion_future_map_transform(*args: Any) -> Any: ...
def legion_future_map_wait_all_results(*args: Any) -> Any: ...
def legion_index_attach_launcher_attach_array_soa(*args: Any) -> Any: ...
def legion_index_attach_launcher_create(*args: Any) -> Any: ...
def legion_index_attach_launcher_destroy(*args: Any) -> Any: ...
def legion_index_attach_launcher_set_deduplicate_across_shards(
    *args: Any,
) -> Any: ...
def legion_index_attach_launcher_set_restricted(*args: Any) -> Any: ...
def legion_index_attach_launcher_set_provenance(*args: Any) -> Any: ...
def legion_index_copy_launcher_add_dst_field(*args: Any) -> Any: ...
def legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_partition(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_dst_region_requirement_logical_partition(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_dst_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_src_field(*args: Any) -> Any: ...
def legion_index_copy_launcher_add_src_indirect_region_requirement_logical_partition(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_src_indirect_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_src_region_requirement_logical_partition(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_add_src_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_create(*args: Any) -> Any: ...
def legion_index_copy_launcher_destroy(*args: Any) -> Any: ...
def legion_index_copy_launcher_execute(*args: Any) -> Any: ...
def legion_index_copy_launcher_set_possible_dst_indirect_out_of_range(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_set_possible_src_indirect_out_of_range(
    *args: Any,
) -> Any: ...
def legion_index_copy_launcher_set_sharding_space(*args: Any) -> Any: ...
def legion_index_copy_launcher_set_provenance(*args: Any) -> Any: ...
def legion_index_copy_launcher_set_mapper_arg(*args: Any) -> Any: ...
def legion_index_fill_launcher_create_from_future_with_domain(
    *args: Any,
) -> Any: ...
def legion_index_fill_launcher_create_from_future_with_space(
    *args: Any,
) -> Any: ...
def legion_index_fill_launcher_destroy(*args: Any) -> Any: ...
def legion_index_fill_launcher_execute(*args: Any) -> Any: ...
def legion_index_fill_launcher_set_sharding_space(*args: Any) -> Any: ...
def legion_index_fill_launcher_set_provenance(*args: Any) -> Any: ...
def legion_index_launcher_add_field(*args: Any) -> Any: ...
def legion_index_launcher_add_flags(*args: Any) -> Any: ...
def legion_index_launcher_add_future(*args: Any) -> Any: ...
def legion_index_launcher_add_point_future(*args: Any) -> Any: ...
def legion_index_launcher_add_region_requirement_logical_partition(
    *args: Any,
) -> Any: ...
def legion_index_launcher_add_region_requirement_logical_partition_reduction(
    *args: Any,
) -> Any: ...
def legion_index_launcher_add_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_index_launcher_add_region_requirement_logical_region_reduction(
    *args: Any,
) -> Any: ...
def legion_index_launcher_create_from_buffer(*args: Any) -> Any: ...
def legion_index_launcher_destroy(*args: Any) -> Any: ...
def legion_index_launcher_execute(*args: Any) -> Any: ...
def legion_index_launcher_execute_deterministic_reduction(
    *args: Any,
) -> Any: ...
def legion_index_launcher_execute_outputs(*args: Any) -> Any: ...
def legion_index_launcher_execute_reduction_and_outputs(*args: Any) -> Any: ...
def legion_index_launcher_set_sharding_space(*args: Any) -> Any: ...
def legion_index_launcher_set_provenance(*args: Any) -> Any: ...
def legion_index_partition_create_by_domain(*args: Any) -> Any: ...
def legion_index_partition_create_by_domain_future_map(*args: Any) -> Any: ...
def legion_index_partition_create_by_image(*args: Any) -> Any: ...
def legion_index_partition_create_by_image_range(*args: Any) -> Any: ...
def legion_index_partition_create_by_preimage(*args: Any) -> Any: ...
def legion_index_partition_create_by_preimage_range(*args: Any) -> Any: ...
def legion_index_partition_create_by_restriction(*args: Any) -> Any: ...
def legion_index_partition_create_by_weights(*args: Any) -> Any: ...
def legion_index_partition_create_by_weights_future_map(*args: Any) -> Any: ...
def legion_index_partition_create_equal(*args: Any) -> Any: ...
def legion_index_partition_destroy_unordered(*args: Any) -> Any: ...
def legion_index_partition_get_color_space(*args: Any) -> Any: ...
def legion_index_partition_get_index_subspace_domain_point(
    *args: Any,
) -> Any: ...
def legion_index_space_create_domain(*args: Any) -> Any: ...
def legion_index_space_destroy_unordered(*args: Any) -> Any: ...
def legion_index_space_get_dim(*args: Any) -> Any: ...
def legion_index_space_get_domain(*args: Any) -> Any: ...
def legion_inline_launcher_add_field(*args: Any) -> Any: ...
def legion_inline_launcher_create_logical_region(*args: Any) -> Any: ...
def legion_inline_launcher_destroy(*args: Any) -> Any: ...
def legion_inline_launcher_execute(*args: Any) -> Any: ...
def legion_inline_launcher_set_provenance(*args: Any) -> Any: ...
def legion_issue_timing_op_seconds(*args: Any) -> Any: ...
def legion_issue_timing_op_microseconds(*args: Any) -> Any: ...
def legion_issue_timing_op_nanoseconds(*args: Any) -> Any: ...
def legion_logical_partition_create(*args: Any) -> Any: ...
def legion_logical_partition_get_logical_subregion(*args: Any) -> Any: ...
def legion_logical_partition_get_subregion(*args: Any) -> Any: ...
def legion_logical_region_create(*args: Any) -> Any: ...
def legion_logical_region_destroy_unordered(*args: Any) -> Any: ...
def legion_machine_create(*args: Any) -> Any: ...
def legion_machine_destroy(*args: Any) -> Any: ...
def legion_memory_query_count(*args: Any) -> Any: ...
def legion_memory_query_create(*args: Any) -> Any: ...
def legion_memory_query_destroy(*args: Any) -> Any: ...
def legion_memory_query_first(*args: Any) -> Any: ...
def legion_memory_query_local_address_space(*args: Any) -> Any: ...
def legion_memory_query_next(*args: Any) -> Any: ...
def legion_memory_query_only_kind(*args: Any) -> Any: ...
def legion_output_requirement_add_field(*args: Any) -> Any: ...
def legion_output_requirement_create(*args: Any) -> Any: ...
def legion_output_requirement_create_region_requirement(*args: Any) -> Any: ...
def legion_output_requirement_destroy(*args: Any) -> Any: ...
def legion_output_requirement_get_parent(*args: Any) -> Any: ...
def legion_output_requirement_get_partition(*args: Any) -> Any: ...
def legion_phase_barrier_create(*args: Any) -> Any: ...
def legion_phase_barrier_advance(*args: Any) -> Any: ...
def legion_phase_barrier_destroy(*args: Any) -> Any: ...
def legion_physical_region_destroy(*args: Any) -> Any: ...
def legion_physical_region_is_mapped(*args: Any) -> Any: ...
def legion_physical_region_wait_until_valid(*args: Any) -> Any: ...
def legion_predicate_true(*args: Any) -> Any: ...
def legion_region_requirement_add_flags(*args: Any) -> Any: ...
def legion_region_requirement_create_logical_partition(*args: Any) -> Any: ...
def legion_region_requirement_create_logical_region(*args: Any) -> Any: ...
def legion_region_requirement_create_logical_region_projection(
    *args: Any,
) -> Any: ...
def legion_region_requirement_destroy(*args: Any) -> Any: ...
def legion_release_launcher_add_field(*args: Any) -> Any: ...
def legion_release_launcher_create(*args: Any) -> Any: ...
def legion_release_launcher_destroy(*args: Any) -> Any: ...
def legion_release_launcher_execute(*args: Any) -> Any: ...
def legion_runtime_issue_execution_fence(*args: Any) -> Any: ...
def legion_runtime_issue_mapping_fence(*args: Any) -> Any: ...
def legion_runtime_remap_region(*args: Any) -> Any: ...
def legion_runtime_unmap_region(*args: Any) -> Any: ...
def legion_task_launcher_add_field(*args: Any) -> Any: ...
def legion_task_launcher_add_flags(*args: Any) -> Any: ...
def legion_task_launcher_add_future(*args: Any) -> Any: ...
def legion_task_launcher_add_region_requirement_logical_region(
    *args: Any,
) -> Any: ...
def legion_task_launcher_add_region_requirement_logical_region_reduction(
    *args: Any,
) -> Any: ...
def legion_task_launcher_create_from_buffer(*args: Any) -> Any: ...
def legion_task_launcher_destroy(*args: Any) -> Any: ...
def legion_task_launcher_execute(*args: Any) -> Any: ...
def legion_task_launcher_execute_outputs(*args: Any) -> Any: ...
def legion_task_launcher_set_local_function_task(*args: Any) -> Any: ...
def legion_task_launcher_set_point(*args: Any) -> Any: ...
def legion_task_launcher_set_sharding_space(*args: Any) -> Any: ...
def legion_task_launcher_set_provenance(*args: Any) -> Any: ...
def legion_unordered_detach_external_resource(*args: Any) -> Any: ...
def legion_runtime_get_runtime(*args: Any) -> Any: ...
def legion_runtime_generate_library_task_ids(*args: Any) -> Any: ...
def legion_runtime_generate_library_mapper_ids(*args: Any) -> Any: ...
def legion_runtime_generate_library_reduction_ids(*args: Any) -> Any: ...
def legion_runtime_generate_library_projection_ids(*args: Any) -> Any: ...
def legion_runtime_generate_library_sharding_ids(*args: Any) -> Any: ...
def legion_runtime_has_context(*args: Any) -> bool: ...
def legion_runtime_local_shard(*args: Any) -> Any: ...
def legion_runtime_select_tunable_value(*args: Any) -> Any: ...
def legion_runtime_total_shards(*args: Any) -> Any: ...
def legion_sharding_functor_invert(*args: Any) -> Any: ...

__all__ = (
    "LEGION_EXTERNAL_INSTANCE",
    "LEGION_COMPUTE_KIND",
    "SYSTEM_MEM",
    "LEGION_REDOP_BASE",
    "LEGION_TYPE_TOTAL",
    "LEGION_READ_ONLY",
    "LEGION_READ_WRITE",
    "LEGION_TYPE_EXCLUSIVE",
    "LEGION_NO_ACCESS",
    "LEGION_WRITE_DISCARD",
    "LEGION_REDOP_KIND_SUM",
    "LEGION_REDOP_KIND_DIFF",
    "LEGION_REDOP_KIND_PROD",
    "LEGION_REDOP_KIND_DIV",
    "LEGION_REDOP_KIND_MAX",
    "LEGION_REDOP_KIND_MIN",
    "LEGION_REDOP_KIND_OR",
    "LEGION_REDOP_KIND_AND",
    "LEGION_REDOP_KIND_XOR",
    "LEGION_TYPE_BOOL",
    "LEGION_TYPE_INT8",
    "LEGION_TYPE_INT16",
    "LEGION_TYPE_INT32",
    "LEGION_TYPE_INT64",
    "LEGION_TYPE_UINT8",
    "LEGION_TYPE_UINT16",
    "LEGION_TYPE_UINT32",
    "LEGION_TYPE_UINT64",
    "LEGION_TYPE_FLOAT16",
    "LEGION_TYPE_FLOAT32",
    "LEGION_TYPE_FLOAT64",
    "LEGION_TYPE_COMPLEX64",
    "LEGION_TYPE_COMPLEX128",
    "LEGATE_CORE_TUNABLE_WINDOW_SIZE",
    "LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE",
    "LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY",
    "LEGATE_CORE_TUNABLE_NUM_PIECES",
    "LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME",
    "legion_acquire_launcher_add_field",
    "legion_acquire_launcher_create",
    "legion_acquire_launcher_destroy",
    "legion_acquire_launcher_execute",
    "legion_argument_map_create",
    "legion_argument_map_destroy",
    "legion_argument_map_from_future_map",
    "legion_argument_map_set_future",
    "legion_argument_map_set_point",
    "legion_attach_external_resources",
    "legion_attach_launcher_add_cpu_soa_field",
    "legion_attach_launcher_create",
    "legion_attach_launcher_destroy",
    "legion_attach_launcher_execute",
    "legion_attach_launcher_set_mapped",
    "legion_attach_launcher_set_restricted",
    "legion_auto_generate_id",
    "legion_context_consensus_match",
    "legion_context_progress_unordered_operations",
    "legion_copy_launcher_add_dst_field",
    "legion_copy_launcher_add_dst_indirect_region_requirement_logical_region",
    "legion_copy_launcher_add_dst_region_requirement_logical_region",
    "legion_copy_launcher_add_dst_region_requirement_logical_region_reduction",
    "legion_copy_launcher_add_src_field",
    "legion_copy_launcher_add_src_indirect_region_requirement_logical_region",
    "legion_copy_launcher_add_src_region_requirement_logical_region",
    "legion_copy_launcher_create",
    "legion_copy_launcher_destroy",
    "legion_copy_launcher_execute",
    "legion_copy_launcher_set_point",
    "legion_copy_launcher_set_possible_dst_indirect_out_of_range",
    "legion_copy_launcher_set_possible_src_indirect_out_of_range",
    "legion_copy_launcher_set_sharding_space",
    "legion_copy_launcher_set_provenance",
    "legion_detach_external_resources",
    "legion_domain_affine_transform_identity",
    "legion_domain_empty",
    "legion_domain_get_volume",
    "legion_domain_is_dense",
    "legion_domain_point_origin",
    "legion_domain_transform_identity",
    "legion_external_resources_destroy",
    "legion_field_allocator_allocate_field",
    "legion_field_allocator_allocate_field_future",
    "legion_field_allocator_create",
    "legion_field_allocator_destroy",
    "legion_field_allocator_free_field",
    "legion_field_allocator_free_field_unordered",
    "legion_field_space_create",
    "legion_field_space_destroy_unordered",
    "legion_fill_launcher_create_from_future",
    "legion_fill_launcher_destroy",
    "legion_fill_launcher_execute",
    "legion_fill_launcher_set_point",
    "legion_fill_launcher_set_sharding_space",
    "legion_fill_launcher_set_provenance",
    "legion_future_destroy",
    "legion_future_from_untyped_pointer",
    "legion_future_get_untyped_pointer",
    "legion_future_get_untyped_size",
    "legion_future_get_void_result",
    "legion_future_is_ready_subscribe",
    "legion_future_map_construct_from_futures",
    "legion_future_map_destroy",
    "legion_future_map_get_future",
    "legion_future_map_reduce",
    "legion_future_map_transform",
    "legion_future_map_wait_all_results",
    "legion_index_attach_launcher_attach_array_soa",
    "legion_index_attach_launcher_create",
    "legion_index_attach_launcher_destroy",
    "legion_index_attach_launcher_set_deduplicate_across_shards",
    "legion_index_attach_launcher_set_restricted",
    "legion_index_copy_launcher_add_dst_field",
    "legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_partition",
    "legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_region",
    "legion_index_copy_launcher_add_dst_region_requirement_logical_partition",
    "legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction",
    "legion_index_copy_launcher_add_dst_region_requirement_logical_region",
    "legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction",
    "legion_index_copy_launcher_add_src_field",
    "legion_index_copy_launcher_add_src_indirect_region_requirement_logical_partition",
    "legion_index_copy_launcher_add_src_indirect_region_requirement_logical_region",
    "legion_index_copy_launcher_add_src_region_requirement_logical_partition",
    "legion_index_copy_launcher_add_src_region_requirement_logical_region",
    "legion_index_copy_launcher_create",
    "legion_index_copy_launcher_destroy",
    "legion_index_copy_launcher_execute",
    "legion_index_copy_launcher_set_possible_dst_indirect_out_of_range",
    "legion_index_copy_launcher_set_possible_src_indirect_out_of_range",
    "legion_index_copy_launcher_set_sharding_space",
    "legion_index_copy_launcher_set_provenance",
    "legion_index_fill_launcher_create_from_future_with_domain",
    "legion_index_fill_launcher_create_from_future_with_space",
    "legion_index_fill_launcher_destroy",
    "legion_index_fill_launcher_execute",
    "legion_index_fill_launcher_set_sharding_space",
    "legion_index_fill_launcher_set_provenance",
    "legion_index_launcher_add_field",
    "legion_index_launcher_add_flags",
    "legion_index_launcher_add_future",
    "legion_index_launcher_add_point_future",
    "legion_index_launcher_add_region_requirement_logical_partition",
    "legion_index_launcher_add_region_requirement_logical_partition_reduction",
    "legion_index_launcher_add_region_requirement_logical_region",
    "legion_index_launcher_add_region_requirement_logical_region_reduction",
    "legion_index_launcher_create_from_buffer",
    "legion_index_launcher_destroy",
    "legion_index_launcher_execute",
    "legion_index_launcher_execute_deterministic_reduction",
    "legion_index_launcher_execute_outputs",
    "legion_index_launcher_execute_reduction_and_outputs",
    "legion_index_launcher_set_sharding_space",
    "legion_index_launcher_set_provenance",
    "legion_index_partition_create_by_domain",
    "legion_index_partition_create_by_domain_future_map",
    "legion_index_partition_create_by_image",
    "legion_index_partition_create_by_image_range",
    "legion_index_partition_create_by_preimage",
    "legion_index_partition_create_by_preimage_range",
    "legion_index_partition_create_by_restriction",
    "legion_index_partition_create_by_weights",
    "legion_index_partition_create_by_weights_future_map",
    "legion_index_partition_create_equal",
    "legion_index_partition_destroy_unordered",
    "legion_index_partition_get_color_space",
    "legion_index_partition_get_index_subspace_domain_point",
    "legion_index_space_create_domain",
    "legion_index_space_destroy_unordered",
    "legion_index_space_get_dim",
    "legion_index_space_get_domain",
    "legion_inline_launcher_add_field",
    "legion_inline_launcher_create_logical_region",
    "legion_inline_launcher_destroy",
    "legion_inline_launcher_execute",
    "legion_logical_partition_create",
    "legion_logical_partition_get_logical_subregion",
    "legion_logical_partition_get_subregion",
    "legion_logical_region_create",
    "legion_logical_region_destroy_unordered",
    "legion_machine_create",
    "legion_machine_destroy",
    "legion_memory_query_count",
    "legion_memory_query_create",
    "legion_memory_query_destroy",
    "legion_memory_query_first",
    "legion_memory_query_local_address_space",
    "legion_memory_query_only_kind",
    "legion_output_requirement_add_field",
    "legion_output_requirement_create",
    "legion_output_requirement_create_region_requirement",
    "legion_output_requirement_destroy",
    "legion_output_requirement_get_parent",
    "legion_output_requirement_get_partition",
    "legion_physical_region_destroy",
    "legion_physical_region_is_mapped",
    "legion_physical_region_wait_until_valid",
    "legion_predicate_true",
    "legion_region_requirement_add_flags",
    "legion_region_requirement_create_logical_partition",
    "legion_region_requirement_create_logical_region",
    "legion_region_requirement_create_logical_region_projection",
    "legion_region_requirement_destroy",
    "legion_release_launcher_add_field",
    "legion_release_launcher_create",
    "legion_release_launcher_destroy",
    "legion_release_launcher_execute",
    "legion_runtime_issue_execution_fence",
    "legion_runtime_issue_mapping_fence",
    "legion_runtime_remap_region",
    "legion_runtime_unmap_region",
    "legion_task_launcher_add_field",
    "legion_task_launcher_add_flags",
    "legion_task_launcher_add_future",
    "legion_task_launcher_add_region_requirement_logical_region",
    "legion_task_launcher_add_region_requirement_logical_region_reduction",
    "legion_task_launcher_create_from_buffer",
    "legion_task_launcher_destroy",
    "legion_task_launcher_execute",
    "legion_task_launcher_execute_outputs",
    "legion_task_launcher_set_local_function_task",
    "legion_task_launcher_set_point",
    "legion_task_launcher_set_sharding_space",
    "legion_task_launcher_set_provenance",
    "legion_unordered_detach_external_resource",
    "legion_runtime_get_runtime",
    "legion_runtime_generate_library_task_ids",
    "legion_runtime_generate_library_mapper_ids",
    "legion_runtime_generate_library_reduction_ids",
    "legion_runtime_generate_library_projection_ids",
    "legion_runtime_generate_library_sharding_ids",
    "legion_runtime_select_tunable_value",
)
