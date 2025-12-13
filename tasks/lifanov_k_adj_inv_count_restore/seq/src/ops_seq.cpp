#include "lifanov_k_adj_inv_count_restore/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "lifanov_k_adj_inv_count_restore/common/include/common.hpp"

namespace lifanov_k_adj_inv_count_restore {

LifanovKAdjacentInversionCountSEQ::LifanovKAdjacentInversionCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LifanovKAdjacentInversionCountSEQ::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool LifanovKAdjacentInversionCountSEQ::PreProcessingImpl() {
  return true;
}

bool LifanovKAdjacentInversionCountSEQ::RunImpl() {
  const auto &data = GetInput();
  std::size_t n = data.size();
  int result = 0;

  for (std::size_t i = 0; i < n - 1; i++) {
    if (data[i] > data[i + 1]) {
      result++;
    }
  }

  GetOutput() = result;
  return true;
}

bool LifanovKAdjacentInversionCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_adj_inv_count_restore
