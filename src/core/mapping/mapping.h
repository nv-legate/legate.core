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

#include <functional>
#include "core/data/scalar.h"
#include "core/mapping/store.h"

/** @defgroup mapping Mapping API
 */

/**
 * @file
 * @brief Legate Mapping API
 */

namespace legate {
namespace mapping {

class Task;

// NOTE: codes are chosen to reflect the precedence between the processor kinds in choosing target
// processors for tasks.

/**
 * @ingroup mapping
 * @brief An enum class for task targets
 */
enum class TaskTarget : int32_t {
  /**
   * @brief Indicates the task be mapped to a GPU
   */
  GPU = 1,
  /**
   * @brief Indicates the task be mapped to an OpenMP processor
   */
  OMP = 2,
  /**
   * @brief Indicates the task be mapped to a CPU
   */
  CPU = 3,
};

/**
 * @ingroup mapping
 * @brief An enum class for store targets
 */
enum class StoreTarget : int32_t {
  /**
   * @brief Indicates the store be mapped to the system memory (host memory)
   */
  SYSMEM = 1,
  /**
   * @brief Indicates the store be mapped to the GPU framebuffer
   */
  FBMEM = 2,
  /**
   * @brief Indicates the store be mapped to the pinned memory for zero-copy GPU accesses
   */
  ZCMEM = 3,
  /**
   * @brief Indicates the store be mapped to the host memory closest to the target CPU
   */
  SOCKETMEM = 4,
};

/**
 * @ingroup mapping
 * @brief An enum class for instance allocation policies
 */
enum class AllocPolicy : int32_t {
  /**
   * @brief Indicates the store can reuse an existing instance
   */
  MAY_ALLOC = 1,
  /**
   * @brief Indicates the store must be mapped to a fresh instance
   */
  MUST_ALLOC = 2,
};

/**
 * @ingroup mapping
 * @brief An enum class for instant layouts
 */
enum class InstLayout : int32_t {
  /**
   * @brief Indicates the store must be mapped to an SOA instance
   */
  SOA = 1,
  /**
   * @brief Indicates the store must be mapped to an AOS instance. No different than `SOA` in a
   * store mapping for a single store
   */
  AOS = 2,
};

/**
 * @ingroup mapping
 * @brief A descriptor for dimension ordering
 */
struct DimOrdering {
 public:
  /**
   * @brief An enum class for kinds of dimension ordering
   */
  enum class Kind : int32_t {
    /**
     * @brief Indicates the instance have C layout (i.e., the last dimension is the leading
     * dimension in the instance)
     */
    C = 1,
    /**
     * @brief Indicates the instance have Fortran layout (i.e., the first dimension is the leading
     * dimension instance)
     */
    FORTRAN = 2,
    /**
     * @brief Indicates the order of dimensions of the instance is manually specified
     */
    CUSTOM = 3,
  };

 public:
  DimOrdering() {}
  DimOrdering(Kind kind, std::vector<int32_t>&& dims = {});

 public:
  /**
   * @brief Creates a C ordering object
   *
   * @return A `DimOrdering` object
   */
  static DimOrdering c_order();
  /**
   * @brief Creates a Fortran ordering object
   *
   * @return A `DimOrdering` object
   */
  static DimOrdering fortran_order();
  /**
   * @brief Creates a custom ordering object
   *
   * @param dims A vector that stores the order of dimensions.
   *
   * @return A `DimOrdering` object
   */
  static DimOrdering custom_order(std::vector<int32_t>&& dims);

 public:
  DimOrdering(const DimOrdering&)            = default;
  DimOrdering& operator=(const DimOrdering&) = default;

 public:
  DimOrdering(DimOrdering&&)            = default;
  DimOrdering& operator=(DimOrdering&&) = default;

 public:
  bool operator==(const DimOrdering&) const;

 public:
  void populate_dimension_ordering(const Store& store,
                                   std::vector<Legion::DimensionKind>& ordering) const;

 public:
  /**
   * @brief Sets the dimension ordering to C
   */
  void set_c_order();
  /**
   * @brief Sets the dimension ordering to Fortran
   */
  void set_fortran_order();
  /**
   * @brief Sets a custom dimension ordering
   *
   * @param dims A vector that stores the order of dimensions.
   */
  void set_custom_order(std::vector<int32_t>&& dims);

 public:
  /**
   * @brief Dimension ordering type
   */
  Kind kind{Kind::C};
  // When relative is true, 'dims' specifies the order of dimensions
  // for the store's local coordinate space, which will be mapped
  // back to the root store's original coordinate space.
  /**
   * @brief If true, the dimension ordering specifies the order of dimensions
   * for the store's current domain, which will be transformed back to the root
   * store's domain.
   */
  bool relative{false};
  /**
   * @brief Dimension list. Used only when the `kind` is `CUSTOM`.
   */
  std::vector<int32_t> dims{};
};

/**
 * @ingroup mapping
 * @brief A descriptor for instance mapping policy
 */
struct InstanceMappingPolicy {
 public:
  /**
   * @brief Target memory type for the instance
   */
  StoreTarget target{StoreTarget::SYSMEM};
  /**
   * @brief Allocation policy
   */
  AllocPolicy allocation{AllocPolicy::MAY_ALLOC};
  /**
   * @brief Instance layout for the instance
   */
  InstLayout layout{InstLayout::SOA};
  /**
   * @brief Dimension ordering for the instance
   */
  DimOrdering ordering{};
  /**
   * @brief If true, the instance must be tight to the store(s); i.e., the instance
   * must not have any extra elements not included in the store(s).
   */
  bool exact{false};

 public:
  InstanceMappingPolicy() {}

 public:
  InstanceMappingPolicy(const InstanceMappingPolicy&)            = default;
  InstanceMappingPolicy& operator=(const InstanceMappingPolicy&) = default;

 public:
  InstanceMappingPolicy(InstanceMappingPolicy&&)            = default;
  InstanceMappingPolicy& operator=(InstanceMappingPolicy&&) = default;

 public:
  bool operator==(const InstanceMappingPolicy&) const;
  bool operator!=(const InstanceMappingPolicy&) const;

 public:
  /**
   * @brief Indicates whether this policy subsumes a given policy
   *
   * Policy `A` subsumes policy `B`, if every instance created under `B` satisfies `A` as well.
   *
   * @param other Policy to check the subsumption against
   *
   * @return true If this policy subsumes `other`
   * @return false Otherwise
   */
  bool subsumes(const InstanceMappingPolicy& other) const;

 private:
  friend class StoreMapping;
  void populate_layout_constraints(const Store& store,
                                   Legion::LayoutConstraintSet& layout_constraints) const;

 public:
  static InstanceMappingPolicy default_policy(StoreTarget target, bool exact = false);
};

/**
 * @ingroup mapping
 * @brief A mapping policy for stores
 */
struct StoreMapping {
 public:
  /**
   * @brief Stores to which the `policy` should be applied
   */
  std::vector<std::reference_wrapper<const Store>> stores{};
  /**
   * @brief Instance mapping policy
   */
  InstanceMappingPolicy policy;

 public:
  StoreMapping() {}

 public:
  StoreMapping(const StoreMapping&)            = default;
  StoreMapping& operator=(const StoreMapping&) = default;

 public:
  StoreMapping(StoreMapping&&)            = default;
  StoreMapping& operator=(StoreMapping&&) = default;

 public:
  bool for_future() const;
  bool for_unbound_store() const;
  const Store& store() const;

 public:
  /**
   * @brief Returns a region requirement index for the stores.
   *
   * Returns an undefined value if the store mapping has more than one store and the stores are
   * mapped to different region requirements.
   *
   * @return Region requirement index
   */
  uint32_t requirement_index() const;
  /**
   * @brief Returns a set of region requirement indices for the stores.
   *
   * @return A set of region requirement indices
   */
  std::set<uint32_t> requirement_indices() const;
  /**
   * @brief Returns the stores' region requirements
   *
   * @return A set of region requirements
   */
  std::set<const Legion::RegionRequirement*> requirements() const;

 private:
  friend class BaseMapper;
  void populate_layout_constraints(Legion::LayoutConstraintSet& layout_constraints) const;

 public:
  /**
   * @brief Creates a `StoreMapping` object following the default mapping poicy
   *
   * @param store Target store for the mapping policy
   * @param target Target memory type for the store
   * @param exact Indicates whether the policy should request an exact instance
   *
   * @return A `StoreMapping` object
   */
  static StoreMapping default_mapping(const Store& store, StoreTarget target, bool exact = false);
};

/**
 * @ingroup mapping
 * @brief An abstract class that defines machine query APIs
 */
class MachineQueryInterface {
 public:
  virtual ~MachineQueryInterface() {}
  /**
   * @brief Returns local CPUs
   *
   * @return A vector of processors
   */
  virtual const std::vector<Processor>& cpus() const = 0;
  /**
   * @brief Returns local GPUs
   *
   * @return A vector of processors
   */
  virtual const std::vector<Processor>& gpus() const = 0;
  /**
   * @brief Returns local OpenMP processors
   *
   * @return A vector of processors
   */
  virtual const std::vector<Processor>& omps() const = 0;
  /**
   * @brief Returns the total number of nodes
   *
   * @return Total number of nodes
   */
  virtual uint32_t total_nodes() const = 0;
};

/**
 * @ingroup mapping
 * @brief An abstract class that defines Legate mapping APIs
 *
 * The APIs give Legate libraries high-level control on task and store mappings
 */
class Mapper {
 public:
  virtual ~Mapper() {}
  /**
   * @brief Sets a machine query interface. This call gives the mapper a chance
   * to cache the machine query interface.
   *
   * @param machine Machine query interface
   */
  virtual void set_machine(const MachineQueryInterface* machine) = 0;
  /**
   * @brief Picks the target processor type for the task
   *
   * @param task Task to map
   * @param options Processor types for which the task has variants
   *
   * @return A target processor type
   */
  virtual TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options) = 0;
  /**
   * @brief Chooses mapping policies for the task's stores.
   *
   * Store mappings can be underspecified; any store of the task that doesn't have a mapping policy
   * will fall back to the default one.
   *
   * @param task Task to map
   * @param options Types of memories to which the stores can be mapped
   *
   * @return A vector of store mappings
   */
  virtual std::vector<StoreMapping> store_mappings(const Task& task,
                                                   const std::vector<StoreTarget>& options) = 0;
  /**
   * @brief Returns a tunable value
   *
   * @param tunable_id a tunable value id
   *
   * @return A tunable value in a `Scalar` object
   */
  virtual Scalar tunable_value(TunableID tunable_id) = 0;
};

}  // namespace mapping
}  // namespace legate
