#include "legate_library.h"
#include "hello_world.h"

namespace hello {

class SquareTask : public Task<SquareTask, SQUARE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& output      = context.outputs()[0];
    auto output_shape = output.shape<1>();
    auto out          = output.write_accessor<float, 1>();

    auto& input      = context.inputs()[0];
    auto input_shape = input.shape<1>();  // should be a 1-Dim array
    auto in          = input.read_accessor<float, 1>();

    auto n = input_shape.volume();
    for (size_t i = 0; i < n; ++i) { out[i] = in[i] * in[i]; }
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::SquareTask::register_variants();
}

}  // namespace
