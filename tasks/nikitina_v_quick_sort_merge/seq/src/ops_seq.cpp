#include "nikitina_v_quick_sort_merge/seq/include/ops_seq.hpp"

#include <vector>

#include "nikitina_v_quick_sort_merge/common/include/common.hpp"

namespace nikitina_v_quick_sort_merge {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TestTaskSEQ::ValidationImpl() {
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool TestTaskSEQ::RunImpl() {
  if (GetOutput().empty()) {
    return true;
  }
  QuickSortImpl(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_quick_sort_merge
