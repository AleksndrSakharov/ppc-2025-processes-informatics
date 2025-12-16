#include "gonozov_l_elem_vec_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"

namespace gonozov_l_elem_vec_sum {

GonozovLElemVecSumSEQ::GonozovLElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0LL;
}

bool GonozovLElemVecSumSEQ::ValidationImpl() {
  return static_cast<int>(GetInput().size()) > 0 && GetOutput() == 0;
  ;
}

bool GonozovLElemVecSumSEQ::PreProcessingImpl() {
  return true;
}

bool GonozovLElemVecSumSEQ::RunImpl() {
  OutType sum = std::accumulate(GetInput().begin(), GetInput().end(), 0LL);

  GetOutput() = sum;
  return true;
}

bool GonozovLElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_elem_vec_sum
