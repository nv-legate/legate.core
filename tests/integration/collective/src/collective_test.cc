#include "collective_test.h"
#include "legate_library.h"

namespace collective {

Legion::Logger logger("legate.collective_tests");

class CollectiveTestTask : public Task<CollectiveTestTask, COLLECTIVE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    // FIXME print rects
  }
};

}  // namespace collective

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  collective::CollectiveTestTask::register_variants();
}

}  // namespace
