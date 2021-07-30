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

#pragma once

#include "legion.h"
// legion.h has to go before these
#include "data/scalar.h"
#include "data/store.h"
#include "legate_c.h"
#include "legate_defines.h"
#include "runtime/runtime.h"
#include "task/task.h"
#include "utilities/deserializer.h"
#include "utilities/dispatch.h"
#include "utilities/type_traits.h"
#include "utilities/typedefs.h"
