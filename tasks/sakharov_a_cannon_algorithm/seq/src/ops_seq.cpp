#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

SakharovACannonAlgorithmSEQ::SakharovACannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmSEQ::ValidationImpl() {
  return HasValidShape(GetInput());
}

bool SakharovACannonAlgorithmSEQ::PreProcessingImpl() {
  const int n = GetInput().size;
  GetOutput().assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  return true;
}

bool SakharovACannonAlgorithmSEQ::RunImpl() {
  const int block_size = SelectBlockSize(GetInput().size);
  GetOutput() = BlockMultiply(GetInput(), block_size);
  return true;
}

bool SakharovACannonAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
