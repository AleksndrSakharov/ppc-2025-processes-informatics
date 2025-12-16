#include "pikhotskiy_r_elem_vec_sum/seq/include/ops_seq.hpp"

#include <utility>
#include <vector>

#include "pikhotskiy_r_elem_vec_sum/common/include/common.hpp"

namespace pikhotskiy_r_elem_vec_sum {
PikhotskiyRElemVecSumSEQ::PikhotskiyRElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PikhotskiyRElemVecSumSEQ::ValidationImpl() {
  const auto &input_tuple = GetInput();
  bool output_ok = (GetOutput() == 0);
  bool size_ok = std::cmp_equal(std::get<1>(input_tuple).size(), std::get<0>(input_tuple));
  return output_ok && size_ok;
}

bool PikhotskiyRElemVecSumSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool PikhotskiyRElemVecSumSEQ::RunImpl() {
  const std::vector<int> &numbers = std::get<1>(GetInput());
  OutType total = 0LL;

  for (const int num : numbers) {
    total += static_cast<OutType>(num);
  }

  GetOutput() = total;
  return true;
}

bool PikhotskiyRElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_elem_vec_sum
