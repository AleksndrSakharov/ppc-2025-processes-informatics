#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

MinValueSEQ::MinValueSEQ(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool MinValueSEQ::ValidationImpl() {
  const auto &mat = this->GetInput();
  if (mat.empty()) {
    return false;
  }
  return std::ranges::all_of(mat, [](const auto &row) { return !row.empty(); });
}

bool MinValueSEQ::PreProcessingImpl() {
  const auto &mat = this->GetInput();
  this->GetOutput().clear();
  this->GetOutput().resize(mat.size());
  return true;
}

bool MinValueSEQ::RunImpl() {
  const auto &mat = this->GetInput();
  auto &out = this->GetOutput();
  for (size_t i = 0; i < mat.size(); ++i) {
    const auto &row = mat[i];
    if (!row.empty()) {
      out[i] = *std::ranges::min_element(row);
    }
  }
  return true;
}

bool MinValueSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_min_value_in_matrix_rows
