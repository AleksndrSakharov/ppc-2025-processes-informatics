#include "lukin_i_elem_vec_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"

namespace lukin_i_elem_vec_sum {

LukinIElemVecSumSEQ::LukinIElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LukinIElemVecSumSEQ::ValidationImpl() {
  return (static_cast<int>(GetInput().size()) != 0);
}

bool LukinIElemVecSumSEQ::PreProcessingImpl() {
  vec_size_ = static_cast<int>(GetInput().size());
  return true;
}

bool LukinIElemVecSumSEQ::RunImpl() {
  OutType sum = std::accumulate(GetInput().begin(), GetInput().end(), 0LL);

  GetOutput() = sum;
  return true;
}

bool LukinIElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
