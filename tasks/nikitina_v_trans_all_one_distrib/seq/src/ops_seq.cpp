#include "nikitina_v_trans_all_one_distrib/seq/include/ops_seq.hpp"

#include "nikitina_v_trans_all_one_distrib/common/include/common.hpp"

namespace nikitina_v_trans_all_one_distrib {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp = in;
  GetInput().swap(tmp);
}

bool TestTaskSEQ::ValidationImpl() {
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  return true;
}

bool TestTaskSEQ::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }
  GetOutput().assign(GetInput().begin(), GetInput().end());
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_trans_all_one_distrib
