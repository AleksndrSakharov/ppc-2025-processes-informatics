#include "pylaeva_s_max_elem_matrix/seq/include/ops_seq.hpp"

#include <algorithm>  // для std::max
#include <cstddef>    // для size_t
#include <limits>     // для std::numeric_limits
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"

namespace pylaeva_s_max_elem_matrix {

PylaevaSMaxElemMatrixSEQ::PylaevaSMaxElemMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<int>::min();
}

bool PylaevaSMaxElemMatrixSEQ::ValidationImpl() {
  const auto rows = static_cast<size_t>(std::get<0>(GetInput()));
  const auto columns = static_cast<size_t>(std::get<1>(GetInput()));
  size_t matrix_size = rows * columns;

  return (matrix_size == std::get<2>(GetInput()).size()) && (rows > 0) && (columns > 0);
}

bool PylaevaSMaxElemMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool PylaevaSMaxElemMatrixSEQ::RunImpl() {
  const auto &matrix_rows = static_cast<size_t>(std::get<0>(GetInput()));
  const auto &matrix_columns = static_cast<size_t>(std::get<1>(GetInput()));
  const auto &matrix_data = std::get<2>(GetInput());

  size_t matrix_size = matrix_rows * matrix_columns;

  if (matrix_data.empty() || matrix_size == 0 || matrix_data.size() != matrix_size) {
    return false;
  }

  int max_element = std::numeric_limits<int>::min();

  for (size_t i = 0; i < matrix_size; ++i) {
    max_element = std::max(matrix_data[i], max_element);
  }

  GetOutput() = max_element;
  return true;
}

bool PylaevaSMaxElemMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
