/* Copyright 2023 NVIDIA Corporation
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

#ifndef __REDUCTION_C_H__
#define __REDUCTION_C_H__

enum OpCode {
  _OP_CODE_BASE = 0,
  BINCOUNT,
  CATEGORIZE,
  HISTOGRAM,
  MATMUL,
  MUL,
  SUM_OVER_AXIS,
  UNIQUE,
};

#endif  // __REDUCTION_C_H__
