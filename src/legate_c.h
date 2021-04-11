/* Copyright 2021 NVIDIA Corporation
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

typedef enum legate_core_task_id_t {
  LEGATE_CORE_INITIALIZE_TASK_ID,
  LEGATE_CORE_FINALIZE_TASK_ID,
  LEGATE_CORE_NUM_TASK_IDS,  // must be last
} legate_core_task_id_t;

typedef enum legate_core_tunable_t {
  LEGATE_CORE_TUNABLE_TOTAL_CPUS = 12345,
  LEGATE_CORE_TUNABLE_TOTAL_GPUS,
} legate_core_tunable_t;

typedef enum legate_core_variant_t {
  LEGATE_NO_VARIANT = 0,
  LEGATE_CPU_VARIANT,
  LEGATE_SSE_VARIANT,
  LEGATE_AVX_VARIANT,
  LEGATE_OMP_VARIANT,
  LEGATE_GPU_VARIANT,
  LEGATE_SUB_CPU_VARIANT,
  LEGATE_SUB_GPU_VARIANT,
  LEGATE_SUB_OMP_VARIANT,
} legate_core_variant_t;

typedef enum legate_core_partition_t {
  PID_ALL_CPUS,
  PID_ALL_GPUS,
  PID_ALL_NUMA,
  PID_ALL_NODES,
  PID_NODE_CPUS,  // sub-partition of PID_ALL_NODES
  PID_NODE_GPUS,  // sub-partition of PID_ALL_NODES
  PID_NODE_NUMA,  // sub-partition of PID_ALL_NODES
} legate_core_partition_t;

// Match these to numpy_field_type_offsets in legate/numpy/config.py
typedef enum legate_core_type_code_t {
  BOOL_LT         = LEGION_TYPE_BOOL,
  INT8_LT         = LEGION_TYPE_INT8,
  INT16_LT        = LEGION_TYPE_INT16,
  INT32_LT        = LEGION_TYPE_INT32,
  INT64_LT        = LEGION_TYPE_INT64,
  UINT8_LT        = LEGION_TYPE_UINT8,
  UINT16_LT       = LEGION_TYPE_UINT16,
  UINT32_LT       = LEGION_TYPE_UINT32,
  UINT64_LT       = LEGION_TYPE_UINT64,
  HALF_LT         = LEGION_TYPE_FLOAT16,
  FLOAT_LT        = LEGION_TYPE_FLOAT32,
  DOUBLE_LT       = LEGION_TYPE_FLOAT64,
  COMPLEX64_LT    = LEGION_TYPE_COMPLEX64,
  COMPLEX128_LT   = LEGION_TYPE_COMPLEX128,
  MAX_TYPE_NUMBER = LEGION_TYPE_TOTAL,  // this must be last
} legate_core_type_code_t;

typedef enum legate_core_resource_t {
  LEGATE_CORE_RESOURCE_CUBLAS,
  LEGATE_CORE_RESOURCE_CUDNN,
  LEGATE_CORE_RESOURCE_CUDF,
  LEGATE_CORE_RESOURCE_CUML,
} legate_core_resource_t;

#ifdef __cplusplus
extern "C" {
#endif

void legate_parse_config(void);
void legate_shutdown(void);

void legate_core_perform_registration(void);

#ifdef __cplusplus
}
#endif

#endif  // __LEGATE_C_H__
