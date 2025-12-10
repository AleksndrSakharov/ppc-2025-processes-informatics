#include "kurpiakov_a_elem_vec_sum/seq/include/ops_seq.hpp"

#include <utility>
#include <vector>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"

namespace kurpiakov_a_elem_vec_sum {
KurpiakovAElemVecSumSEQ::KurpiakovAElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KurpiakovAElemVecSumSEQ::ValidationImpl() {
  bool res = (GetOutput() == 0) && (std::cmp_equal((std::get<1>(GetInput()).size()), std::get<0>(GetInput())));
  return res;
}

bool KurpiakovAElemVecSumSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KurpiakovAElemVecSumSEQ::RunImpl() {
  std::vector<int> vec = std::get<1>(GetInput());
  OutType res = 0LL;
  for (const int &it : vec) {
    res += static_cast<OutType>(it);
  }
  GetOutput() = res;
  return true;
}

bool KurpiakovAElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_elem_vec_sum
