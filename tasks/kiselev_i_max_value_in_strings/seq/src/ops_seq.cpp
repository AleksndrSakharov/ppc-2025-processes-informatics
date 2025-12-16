#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"

namespace kiselev_i_max_value_in_strings {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  GetOutput().clear();
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  return true;
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput();
  GetOutput().resize(matrix.size());
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &out_vector = GetOutput();

  std::size_t row_idx = 0;
  for (const auto &row : matrix) {
    if (row.empty()) {
      // пустая строка → минимальное значение int
      out_vector[row_idx++] = std::numeric_limits<int>::min();
    } else {
      int tmp_max = row[0];
      for (int val : row) {
        tmp_max = std::max(val, tmp_max);
      }
      out_vector[row_idx++] = tmp_max;
    }
  }
  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
