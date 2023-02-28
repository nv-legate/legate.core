#include "hello_world.h"
#include "legate_library.h"

namespace hello {

class IotaTask : public Task<IotaTask, IOTA> {
public:
  static void cpu_variant(legate::TaskContext &context) {

    legate::Store &output = context.outputs()[0];
    legate::Rect<1> output_shape = output.shape<1>();
    auto out = output.write_accessor<float, 1>();

    logger.info() << "Iota task [" << output_shape.lo << "," << output_shape.hi
                  << "]";

    // i is a global index for the complete array
    for (size_t i = output_shape.lo; i <= output_shape.hi; ++i) {
      out[i] = i + 1;
    }
  }
};

} // namespace hello

namespace // unnamed
{

static void __attribute__((constructor)) register_tasks(void) {
  hello::IotaTask::register_variants();
}

} // namespace
