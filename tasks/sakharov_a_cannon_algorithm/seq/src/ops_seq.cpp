#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"

#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

SakharovACannonAlgorithmSEQ::SakharovACannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmSEQ::ValidationImpl() {
  return std::get<0>(GetInput()) == 0;
}

bool SakharovACannonAlgorithmSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SakharovACannonAlgorithmSEQ::RunImpl() {
  GetOutput() = std::get<1>(GetInput());
  return true;
}

bool SakharovACannonAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
