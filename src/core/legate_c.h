/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef __LEGATE_C_H__
#define __LEGATE_C_H__

#ifndef LEGATE_USE_PYTHON_CFFI
#include "legion/legion_config.h"
//
#include <cstdint>
#endif

typedef enum legate_core_task_id_t {
  LEGATE_CORE_EXTRACT_SCALAR_TASK_ID,
  LEGATE_CORE_INIT_NCCL_ID_TASK_ID,
  LEGATE_CORE_INIT_NCCL_TASK_ID,
  LEGATE_CORE_FINALIZE_NCCL_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_TASK_ID,
  LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID,
  LEGATE_CORE_NUM_TASK_IDS,  // must be last
} legate_core_task_id_t;

typedef enum legate_core_proj_id_t {
  // local id 0 always maps to the identity projection (global id 0)
  LEGATE_CORE_DELINEARIZE_PROJ_ID      = 2,
  LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID = 10,
  LEGATE_CORE_MAX_FUNCTOR_ID           = 3000000,
} legate_core_proj_id_t;

typedef enum legate_core_shard_id_t {
  LEGATE_CORE_TOPLEVEL_TASK_SHARD_ID = 0,
  LEGATE_CORE_LINEARIZE_SHARD_ID     = 1,
  // All sharding functors starting from LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR should match the
  // projection functor of the same id. The sharding functor limit is thus the same as the
  // projection functor limit.
} legate_core_shard_id_t;

typedef enum legate_core_tunable_t {
  LEGATE_CORE_TUNABLE_TOTAL_CPUS = 12345,
  LEGATE_CORE_TUNABLE_TOTAL_OMPS,
  LEGATE_CORE_TUNABLE_TOTAL_GPUS,
  LEGATE_CORE_TUNABLE_NUM_PIECES,
  LEGATE_CORE_TUNABLE_NUM_NODES,
  LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM,
  LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME,
  LEGATE_CORE_TUNABLE_WINDOW_SIZE,
  LEGATE_CORE_TUNABLE_MAX_PENDING_EXCEPTIONS,
  LEGATE_CORE_TUNABLE_PRECISE_EXCEPTION_TRACE,
  LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE,
  LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY,
  LEGATE_CORE_TUNABLE_MAX_LRU_LENGTH,
  LEGATE_CORE_TUNABLE_NCCL_NEEDS_BARRIER,
} legate_core_tunable_t;

typedef enum legate_core_variant_t {
  LEGATE_NO_VARIANT = 0,
  LEGATE_CPU_VARIANT,
  LEGATE_GPU_VARIANT,
  LEGATE_OMP_VARIANT,
} legate_core_variant_t;

// Match these to numpy_field_type_offsets in legate/numpy/config.py
typedef enum legate_core_type_code_t {
  // Buil-in primitive types
  BOOL_LT       = LEGION_TYPE_BOOL,
  INT8_LT       = LEGION_TYPE_INT8,
  INT16_LT      = LEGION_TYPE_INT16,
  INT32_LT      = LEGION_TYPE_INT32,
  INT64_LT      = LEGION_TYPE_INT64,
  UINT8_LT      = LEGION_TYPE_UINT8,
  UINT16_LT     = LEGION_TYPE_UINT16,
  UINT32_LT     = LEGION_TYPE_UINT32,
  UINT64_LT     = LEGION_TYPE_UINT64,
  FLOAT16_LT    = LEGION_TYPE_FLOAT16,
  FLOAT32_LT    = LEGION_TYPE_FLOAT32,
  FLOAT64_LT    = LEGION_TYPE_FLOAT64,
  COMPLEX64_LT  = LEGION_TYPE_COMPLEX64,
  COMPLEX128_LT = LEGION_TYPE_COMPLEX128,
  // Compound types
  FIXED_ARRAY_LT,
  STRUCT_LT,
  // Variable size types
  STRING_LT,
  INVALID_LT = -1,
} legate_core_type_code_t;

typedef enum legate_core_reduction_op_kind_t {
  ADD_LT = LEGION_REDOP_KIND_SUM,
  SUB_LT = LEGION_REDOP_KIND_DIFF,
  MUL_LT = LEGION_REDOP_KIND_PROD,
  DIV_LT = LEGION_REDOP_KIND_DIV,
  MAX_LT = LEGION_REDOP_KIND_MAX,
  MIN_LT = LEGION_REDOP_KIND_MIN,
  OR_LT  = LEGION_REDOP_KIND_OR,
  AND_LT = LEGION_REDOP_KIND_AND,
  XOR_LT = LEGION_REDOP_KIND_XOR,
} legate_core_reduction_op_kind_t;

typedef enum legate_core_transform_t {
  LEGATE_CORE_TRANSFORM_SHIFT = 100,
  LEGATE_CORE_TRANSFORM_PROMOTE,
  LEGATE_CORE_TRANSFORM_PROJECT,
  LEGATE_CORE_TRANSFORM_TRANSPOSE,
  LEGATE_CORE_TRANSFORM_DELINEARIZE,
} legate_core_transform_t;

typedef enum legate_core_mapping_tag_t {
  LEGATE_CORE_KEY_STORE_TAG              = 1,
  LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG = 2,
  LEGATE_CORE_TREE_REDUCE_TAG            = 3,
  LEGATE_CORE_JOIN_EXCEPTION_TAG         = 4,
} legate_core_mapping_tag_t;

typedef enum legate_core_reduction_op_id_t {
  LEGATE_CORE_JOIN_EXCEPTION_OP   = 0,
  LEGATE_CORE_MAX_REDUCTION_OP_ID = 1,
} legate_core_reduction_op_id_t;

#ifdef __cplusplus
extern "C" {
#endif

void legate_parse_config(void);
void legate_shutdown(void);

void legate_core_perform_registration(void);

void legate_register_affine_projection_functor(
  int32_t, int32_t, int32_t*, int32_t*, int32_t*, legion_projection_id_t);

void legate_create_sharding_functor_using_projection(
  legion_sharding_id_t, legion_projection_id_t, uint32_t, uint32_t, uint32_t, uint32_t);

// TODO: the return type should be legion_point_transform_functor_t
void* legate_linearizing_point_transform_functor();

void legate_cpucoll_finalize(void);

int legate_cpucoll_initcomm(void);

#ifdef __cplusplus
}
#endif

#endif  // __LEGATE_C_H__
