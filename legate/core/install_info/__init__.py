# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See the LICENSE file for details.
#

# IMPORTANT:
#   * header.py.in is used as an input to string.format()
#   * header.py is a generated file and should not be modified by hand
from __future__ import annotations

libpath: str = "/opt/legate/core/_skbuild/linux-x86_64-3.9/cmake-build/legate_core-cpp/lib"
header: str = """typedef enum legate_core_task_id_t {
  LEGATE_CORE_EXTRACT_SCALAR_TASK_ID,
  LEGATE_CORE_INIT_NCCL_ID_TASK_ID,
  LEGATE_CORE_INIT_NCCL_TASK_ID,
  LEGATE_CORE_FINALIZE_NCCL_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_ID_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_TASK_ID,
  LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID,
  LEGATE_CORE_NUM_TASK_IDS,
} legate_core_task_id_t;
typedef enum legate_core_proj_id_t {
  LEGATE_CORE_DELINEARIZE_PROJ_ID = 2,
  LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID = 10,
  LEGATE_CORE_MAX_FUNCTOR_ID = 3000000,
} legate_core_proj_id_t;
typedef enum legate_core_shard_id_t {
  LEGATE_CORE_TOPLEVEL_TASK_SHARD_ID = 0,
  LEGATE_CORE_LINEARIZE_SHARD_ID = 1,
} legate_core_shard_id_t;
typedef enum legate_core_tunable_t {
  LEGATE_CORE_TUNABLE_TOTAL_CPUS = 12345,
  LEGATE_CORE_TUNABLE_TOTAL_GPUS,
  LEGATE_CORE_TUNABLE_NUM_PIECES,
  LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME,
  LEGATE_CORE_TUNABLE_WINDOW_SIZE,
  LEGATE_CORE_TUNABLE_MAX_PENDING_EXCEPTIONS,
  LEGATE_CORE_TUNABLE_PRECISE_EXCEPTION_TRACE,
  LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE,
  LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY,
  LEGATE_CORE_TUNABLE_NCCL_NEEDS_BARRIER,
} legate_core_tunable_t;
typedef enum legate_core_variant_t {
  LEGATE_NO_VARIANT = 0,
  LEGATE_CPU_VARIANT,
  LEGATE_GPU_VARIANT,
  LEGATE_OMP_VARIANT,
} legate_core_variant_t;
typedef enum legate_core_type_code_t {
  BOOL_LT = LEGION_TYPE_BOOL,
  INT8_LT = LEGION_TYPE_INT8,
  INT16_LT = LEGION_TYPE_INT16,
  INT32_LT = LEGION_TYPE_INT32,
  INT64_LT = LEGION_TYPE_INT64,
  UINT8_LT = LEGION_TYPE_UINT8,
  UINT16_LT = LEGION_TYPE_UINT16,
  UINT32_LT = LEGION_TYPE_UINT32,
  UINT64_LT = LEGION_TYPE_UINT64,
  HALF_LT = LEGION_TYPE_FLOAT16,
  FLOAT_LT = LEGION_TYPE_FLOAT32,
  DOUBLE_LT = LEGION_TYPE_FLOAT64,
  COMPLEX64_LT = LEGION_TYPE_COMPLEX64,
  COMPLEX128_LT = LEGION_TYPE_COMPLEX128,
  STRING_LT = COMPLEX128_LT + 1,
  MAX_TYPE_NUMBER,
} legate_core_type_code_t;
typedef enum legate_core_transform_t {
  LEGATE_CORE_TRANSFORM_SHIFT = 100,
  LEGATE_CORE_TRANSFORM_PROMOTE,
  LEGATE_CORE_TRANSFORM_PROJECT,
  LEGATE_CORE_TRANSFORM_TRANSPOSE,
  LEGATE_CORE_TRANSFORM_DELINEARIZE,
} legate_core_transform_t;
typedef enum legate_core_mapping_tag_t {
  LEGATE_CORE_KEY_STORE_TAG = 1,
  LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG = 2,
  LEGATE_CORE_TREE_REDUCE_TAG = 3,
} legate_core_mapping_tag_t;
typedef enum legate_core_reduction_op_id_t {
  LEGATE_CORE_JOIN_EXCEPTION_OP = 0,
  LEGATE_CORE_MAX_REDUCTION_OP_ID = 1,
} legate_core_reduction_op_id_t;
void legate_parse_config(void);
void legate_shutdown(void);
void legate_core_perform_registration(void);
void legate_register_affine_projection_functor(
  int32_t, int32_t, int32_t*, int32_t*, int32_t*, legion_projection_id_t);
void legate_create_sharding_functor_using_projection(legion_sharding_id_t, legion_projection_id_t);
void* legate_linearizing_point_transform_functor();
void legate_cpucoll_finalize(void);
"""
