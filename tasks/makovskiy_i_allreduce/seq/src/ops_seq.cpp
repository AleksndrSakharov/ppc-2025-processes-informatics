#include "makovskiy_i_allreduce/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "makovskiy_i_allreduce/common/include/common.hpp"

namespace makovskiy_i_allreduce {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool TestTaskSEQ::ValidationImpl() {
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  this->GetOutput().resize(1);
  return true;
}

bool TestTaskSEQ::RunImpl() {
  const auto &input = this->GetInput();
  if (input.empty()) {
    this->GetOutput()[0] = 0;
    return true;
  }
  int res = std::accumulate(input.begin(), input.end(), 0);
  this->GetOutput()[0] = res;
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_allreduce
