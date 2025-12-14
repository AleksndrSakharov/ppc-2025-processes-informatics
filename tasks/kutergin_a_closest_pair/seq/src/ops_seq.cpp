#include "kutergin_a_closest_pair/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdlib>
#include <ranges>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"

namespace kutergin_a_closest_pair {

KuterginAClosestPairSEQ::KuterginAClosestPairSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -1;
}

bool KuterginAClosestPairSEQ::ValidationImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::PreProcessingImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::RunImpl() {
  const auto &v = GetInput();

  if (v.size() < 2) {
    GetOutput() = -1;
    return true;
  }

  auto idx_range = std::views::iota(size_t{0}, v.size() - 1);

  auto comparator = [&](size_t i, size_t j) {
    int diff_i = std::abs(v[i] - v[i + 1]);
    int diff_j = std::abs(v[j] - v[j + 1]);
    return (diff_i < diff_j) || (diff_i == diff_j && i < j);
  };

  auto min_it = std::ranges::min_element(idx_range, comparator);
  int min_idx = static_cast<int>(*min_it);

  GetOutput() = min_idx;
  return true;
}

bool KuterginAClosestPairSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_closest_pair
