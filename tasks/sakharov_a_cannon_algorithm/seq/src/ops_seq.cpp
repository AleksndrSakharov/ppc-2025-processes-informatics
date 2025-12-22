#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

SakharovACannonAlgorithmSEQ::SakharovACannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmSEQ::ValidationImpl() {
  return IsValidInput(GetInput());
}

bool SakharovACannonAlgorithmSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  auto out_size = static_cast<std::size_t>(input.rows_a) * static_cast<std::size_t>(input.cols_b);
  GetOutput().assign(out_size, 0.0);
  return true;
}

bool SakharovACannonAlgorithmSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int m = input.rows_a;
  const int k = input.cols_a;
  const int n = input.cols_b;

  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      double a_val = input.a[Idx(k, i, p)];
      for (int j = 0; j < n; ++j) {
        output[Idx(n, i, j)] += a_val * input.b[Idx(n, p, j)];
      }
    }
  }

  return true;
}

bool SakharovACannonAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
