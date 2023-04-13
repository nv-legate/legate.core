#include "collective_test.h"
#include "legate_library.h"

namespace integration {

Legion::Logger logger("legate.integration_tests");

class CollectiveTestTask : public Task<CollectiveTestTask, COLLECTIVE> {
public:
  static void cpu_variant(legate::TaskContext &context) {
    // FIXME print rects
  }
};

} // namespace integration

namespace // unnamed
{

static void __attribute__((constructor)) register_tasks(void) {
  integration::CollectiveTestTask::register_variants();
}

} // namespace
