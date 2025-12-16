#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"

namespace kruglova_a_max_diff_adjacent {

KruglovaAMaxDiffAdjacentSEQ::KruglovaAMaxDiffAdjacentSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0F;
}

bool KruglovaAMaxDiffAdjacentSEQ::ValidationImpl() {
  return true;
}

bool KruglovaAMaxDiffAdjacentSEQ::PreProcessingImpl() {
  return true;
}

bool KruglovaAMaxDiffAdjacentSEQ::RunImpl() {
  const auto &vec = this->GetInput();
  auto &out = this->GetOutput();

  if (vec.size() < 2) {
    out = 0.0F;
    return true;
  }

  float max_diff = std::abs(vec[1] - vec[0]);
  for (size_t i = 1; i < vec.size(); ++i) {
    float diff = std::abs(vec[i] - vec[i - 1]);
    max_diff = std::max(diff, max_diff);
  }
  out = max_diff;
  return true;
}

bool KruglovaAMaxDiffAdjacentSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_max_diff_adjacent
