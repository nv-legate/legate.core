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

#pragma once

#include "core/type/type_info.h"

/**
 * @file
 * @brief Definitions for dispatch routines
 */
namespace legate {

template <int DIM>
struct inner_type_dispatch_fn {
  template <typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(Type::Code code, Functor f, Fnargs&&... args)
  {
    switch (code) {
      case Type::Code::BOOL: {
        return f.template operator()<Type::Code::BOOL, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::INT8: {
        return f.template operator()<Type::Code::INT8, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::INT16: {
        return f.template operator()<Type::Code::INT16, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::INT32: {
        return f.template operator()<Type::Code::INT32, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::INT64: {
        return f.template operator()<Type::Code::INT64, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::UINT8: {
        return f.template operator()<Type::Code::UINT8, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::UINT16: {
        return f.template operator()<Type::Code::UINT16, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::UINT32: {
        return f.template operator()<Type::Code::UINT32, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::UINT64: {
        return f.template operator()<Type::Code::UINT64, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::FLOAT16: {
        return f.template operator()<Type::Code::FLOAT16, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::FLOAT32: {
        return f.template operator()<Type::Code::FLOAT32, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::FLOAT64: {
        return f.template operator()<Type::Code::FLOAT64, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::COMPLEX64: {
        return f.template operator()<Type::Code::COMPLEX64, DIM>(std::forward<Fnargs>(args)...);
      }
      case Type::Code::COMPLEX128: {
        return f.template operator()<Type::Code::COMPLEX128, DIM>(std::forward<Fnargs>(args)...);
      }
      default: break;
    }
    assert(false);
    return f.template operator()<Type::Code::BOOL, DIM>(std::forward<Fnargs>(args)...);
  }
};

template <int DIM>
struct inner_dim_dispatch_fn {
  template <typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(int dim, Functor f, Fnargs&&... args)
  {
    switch (dim) {
      case 1: {
        return f.template operator()<DIM, 1>(std::forward<Fnargs>(args)...);
      }
#if LEGATE_MAX_DIM >= 2
      case 2: {
        return f.template operator()<DIM, 2>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 3
      case 3: {
        return f.template operator()<DIM, 3>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 4
      case 4: {
        return f.template operator()<DIM, 4>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 5
      case 5: {
        return f.template operator()<DIM, 5>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 6
      case 6: {
        return f.template operator()<DIM, 6>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 7
      case 7: {
        return f.template operator()<DIM, 7>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 8
      case 8: {
        return f.template operator()<DIM, 8>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 9
      case 9: {
        return f.template operator()<DIM, 9>(std::forward<Fnargs>(args)...);
      }
#endif
    }
    assert(false);
    return f.template operator()<DIM, 1>(std::forward<Fnargs>(args)...);
  }
};

/**
 * @ingroup util
 * @brief Converts the runtime dimension and type code into compile time constants and
 * invokes the functor with them
 *
 * The functor's `operator()` should take a dimension and a type code as template parameters.
 *
 * @param dim Dimension
 * @param code Type code
 * @param f Functor to dispatch
 * @param args Extra arguments to the functor
 *
 * @return The functor's return value
 */
template <typename Functor, typename... Fnargs>
constexpr decltype(auto) double_dispatch(int dim, Type::Code code, Functor f, Fnargs&&... args)
{
  switch (dim) {
#if LEGATE_MAX_DIM >= 1
    case 1: {
      return inner_type_dispatch_fn<1>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 2
    case 2: {
      return inner_type_dispatch_fn<2>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 3
    case 3: {
      return inner_type_dispatch_fn<3>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 4
    case 4: {
      return inner_type_dispatch_fn<4>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 5
    case 5: {
      return inner_type_dispatch_fn<5>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 6
    case 6: {
      return inner_type_dispatch_fn<6>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 7
    case 7: {
      return inner_type_dispatch_fn<7>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 8
    case 8: {
      return inner_type_dispatch_fn<8>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 9
    case 9: {
      return inner_type_dispatch_fn<9>{}(code, f, std::forward<Fnargs>(args)...);
    }
#endif
  }
  assert(false);
  return inner_type_dispatch_fn<1>{}(code, f, std::forward<Fnargs>(args)...);
}

/**
 * @ingroup util
 * @brief Converts the runtime dimensions into compile time constants and invokes
 * the functor with them
 *
 * The functor's `operator()` should take exactly two integers as template parameters.
 *
 * @param dim1 First dimension
 * @param dim2 Second dimension
 * @param f Functor to dispatch
 * @param args Extra arguments to the functor
 *
 * @return The functor's return value
 */
template <typename Functor, typename... Fnargs>
constexpr decltype(auto) double_dispatch(int dim1, int dim2, Functor f, Fnargs&&... args)
{
  switch (dim1) {
#if LEGATE_MAX_DIM >= 1
    case 1: {
      return inner_dim_dispatch_fn<1>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 2
    case 2: {
      return inner_dim_dispatch_fn<2>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 3
    case 3: {
      return inner_dim_dispatch_fn<3>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 4
    case 4: {
      return inner_dim_dispatch_fn<4>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 5
    case 5: {
      return inner_dim_dispatch_fn<5>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 6
    case 6: {
      return inner_dim_dispatch_fn<6>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 7
    case 7: {
      return inner_dim_dispatch_fn<7>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 8
    case 8: {
      return inner_dim_dispatch_fn<8>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 9
    case 9: {
      return inner_dim_dispatch_fn<9>{}(dim2, f, std::forward<Fnargs>(args)...);
    }
#endif
  }
  assert(false);
  return inner_dim_dispatch_fn<1>{}(dim2, f, std::forward<Fnargs>(args)...);
}

/**
 * @ingroup util
 * @brief Converts the runtime dimension into a compile time constant and invokes
 * the functor with it
 *
 * The functor's `operator()` should take an integer as its sole template parameter.
 *
 * @param dim Dimension
 * @param f Functor to dispatch
 * @param args Extra arguments to the functor
 *
 * @return The functor's return value
 */
template <typename Functor, typename... Fnargs>
constexpr decltype(auto) dim_dispatch(int dim, Functor f, Fnargs&&... args)
{
  switch (dim) {
#if LEGATE_MAX_DIM >= 1
    case 1: {
      return f.template operator()<1>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 2
    case 2: {
      return f.template operator()<2>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 3
    case 3: {
      return f.template operator()<3>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 4
    case 4: {
      return f.template operator()<4>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 5
    case 5: {
      return f.template operator()<5>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 6
    case 6: {
      return f.template operator()<6>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 7
    case 7: {
      return f.template operator()<7>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 8
    case 8: {
      return f.template operator()<8>(std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 9
    case 9: {
      return f.template operator()<9>(std::forward<Fnargs>(args)...);
    }
#endif
  }
  assert(false);
  return f.template operator()<1>(std::forward<Fnargs>(args)...);
}

/**
 * @ingroup util
 * @brief Converts the runtime type code into a compile time constant and invokes
 * the functor with it
 *
 * The functor's `operator()` should take a type code as its sole template parameter.
 *
 * @param code Type code
 * @param f Functor to dispatch
 * @param args Extra arguments to the functor
 *
 * @return The functor's return value
 */
template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch(Type::Code code, Functor f, Fnargs&&... args)
{
  switch (code) {
    case Type::Code::BOOL: {
      return f.template operator()<Type::Code::BOOL>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::INT8: {
      return f.template operator()<Type::Code::INT8>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::INT16: {
      return f.template operator()<Type::Code::INT16>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::INT32: {
      return f.template operator()<Type::Code::INT32>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::INT64: {
      return f.template operator()<Type::Code::INT64>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::UINT8: {
      return f.template operator()<Type::Code::UINT8>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::UINT16: {
      return f.template operator()<Type::Code::UINT16>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::UINT32: {
      return f.template operator()<Type::Code::UINT32>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::UINT64: {
      return f.template operator()<Type::Code::UINT64>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::FLOAT16: {
      return f.template operator()<Type::Code::FLOAT16>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::FLOAT32: {
      return f.template operator()<Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::FLOAT64: {
      return f.template operator()<Type::Code::FLOAT64>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::COMPLEX64: {
      return f.template operator()<Type::Code::COMPLEX64>(std::forward<Fnargs>(args)...);
    }
    case Type::Code::COMPLEX128: {
      return f.template operator()<Type::Code::COMPLEX128>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<Type::Code::BOOL>(std::forward<Fnargs>(args)...);
}

}  // namespace legate
