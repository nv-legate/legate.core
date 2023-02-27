#include "legate_library.h"
#include "hello_world.h"

namespace hello {

class SumTask : public Task<SumTask, SUM> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input      = context.inputs()[0];
    auto input_shape = input.shape<1>();  // should be a 1-Dim array
    auto in          = input.read_accessor<float, 1>();

    auto n      = input_shape.volume();
    float total = 0;
    for (size_t i = 0; i < n; ++i) { total += in[i]; }

    using Reduce = Legion::SumReduction<float>;
    auto sum     = context.reductions()[0].reduce_accessor<Reduce, true, 1>();
    sum.reduce(0, total);
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::SumTask::register_variants();
}

}  // namespace
