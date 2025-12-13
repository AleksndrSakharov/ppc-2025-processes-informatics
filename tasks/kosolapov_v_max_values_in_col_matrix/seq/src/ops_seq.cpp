#include "kosolapov_v_max_values_in_col_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixSEQ::KosolapovVMaxValuesInColMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }
  for (size_t i = 0; i < matrix.size() - 1; i++) {
    if (matrix[i].size() != matrix[i + 1].size()) {
      return false;
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput()[0].size());
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  for (size_t i = 0; i < matrix[0].size(); i++) {
    int temp_max = matrix[0][i];
    for (const auto &row : matrix) {
      temp_max = std::max(row[i], temp_max);
    }
    GetOutput()[i] = temp_max;
  }
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kosolapov_v_max_values_in_col_matrix
