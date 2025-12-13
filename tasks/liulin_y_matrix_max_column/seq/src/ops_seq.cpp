#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"

namespace liulin_y_matrix_max_column {

LiulinYMatrixMaxColumnSEQ::LiulinYMatrixMaxColumnSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool LiulinYMatrixMaxColumnSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.empty() || in[0].empty()) {
    return true;
  }

  const std::size_t cols = in[0].size();

  for (const auto &row : in) {
    if (row.size() != cols) {
      return true;
    }
  }

  return GetOutput().empty();
}

bool LiulinYMatrixMaxColumnSEQ::PreProcessingImpl() {
  const auto &in = GetInput();

  if (in.empty() || in[0].empty()) {
    GetOutput().clear();
    return true;
  }

  const std::size_t cols = in[0].size();
  GetOutput().assign(cols, std::numeric_limits<int>::min());
  return true;
}

bool LiulinYMatrixMaxColumnSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &out = GetOutput();

  if (matrix.empty() || matrix[0].empty()) {
    return true;
  }
  const int rows = static_cast<int>(matrix.size());
  const int cols = static_cast<int>(matrix[0].size());

  for (int col_idx = 0; col_idx < cols; ++col_idx) {
    std::vector<int> column(rows);
    for (int row = 0; row < rows; ++row) {
      column[row] = matrix[row][col_idx];
    }

    int size = rows;
    std::vector<int> temp = column;

    while (size > 1) {
      int new_size = 0;
      for (int i = 0; i < size; i += 2) {
        temp[new_size] = (i + 1 < size) ? std::max(temp[i], temp[i + 1]) : temp[i];
        ++new_size;
      }
      size = new_size;
    }

    out[col_idx] = temp[0];
  }

  return true;
}

bool LiulinYMatrixMaxColumnSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_matrix_max_column
