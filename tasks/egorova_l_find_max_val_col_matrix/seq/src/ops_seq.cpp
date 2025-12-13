#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"

namespace egorova_l_find_max_val_col_matrix {

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

EgorovaLFindMaxValColMatrixSEQ::EgorovaLFindMaxValColMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(0);
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

bool EgorovaLFindMaxValColMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return true;
  }

  if (matrix[0].empty()) {
    return true;
  }

  const std::size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool EgorovaLFindMaxValColMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool EgorovaLFindMaxValColMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();

  // Оставляем только проверку на пустоту
  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  const std::size_t rows = matrix.size();
  const std::size_t cols = matrix[0].size();
  std::vector<int> result(cols, std::numeric_limits<int>::min());

  for (std::size_t jj = 0; jj < cols; ++jj) {
    for (std::size_t ii = 0; ii < rows; ++ii) {
      result[jj] = std::max(matrix[ii][jj], result[jj]);
    }
  }

  GetOutput() = result;
  return true;
}

bool EgorovaLFindMaxValColMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_find_max_val_col_matrix
