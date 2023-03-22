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

#ifndef __LEGATE_IO_C_H__
#define __LEGATE_IO_C_H__

enum LegateIOOpCode {
  _OP_CODE_BASE = 0,
  READ_EVEN_TILES,
  READ_FILE,
  READ_UNEVEN_TILES,
  WRITE_EVEN_TILES,
  WRITE_FILE,
  WRITE_UNEVEN_TILES,
};

#endif  // __LEGATE_IO_C_H__
