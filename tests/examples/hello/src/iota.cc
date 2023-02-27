#include "legate_library.h"
#include "hello_world.h"

namespace hello {

class IotaTask : public Task<IotaTask, IOTA> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& output      = context.outputs()[0];
    auto output_shape = output.shape<1>();
    auto out          = output.write_accessor<float, 1>();
    auto n            = output_shape.volume();
    for (size_t i = 0; i < n; ++i) { out[i] = i + 1; }
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::IotaTask::register_variants();
}

}  // namespace
