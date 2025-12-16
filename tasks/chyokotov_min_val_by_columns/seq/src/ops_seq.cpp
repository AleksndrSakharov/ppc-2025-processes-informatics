#include "chyokotov_min_val_by_columns/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"

namespace chyokotov_min_val_by_columns {

ChyokotovMinValByColumnsSEQ::ChyokotovMinValByColumnsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool ChyokotovMinValByColumnsSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return true;
  }

  size_t length_row = input[0].size();
  return std::ranges::all_of(input, [length_row](const auto &row) { return row.size() == length_row; });
}

bool ChyokotovMinValByColumnsSEQ::PreProcessingImpl() {
  if (GetInput().empty()) {
    return true;
  }
  GetOutput().resize(GetInput()[0].size(), INT_MAX);
  return true;
}

bool ChyokotovMinValByColumnsSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  auto &output = GetOutput();

  for (size_t i = 0; i < output.size(); i++) {
    for (const auto &row : matrix) {
      output[i] = std::min(output[i], row[i]);
    }
  }

  return true;
}

bool ChyokotovMinValByColumnsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_min_val_by_columns
