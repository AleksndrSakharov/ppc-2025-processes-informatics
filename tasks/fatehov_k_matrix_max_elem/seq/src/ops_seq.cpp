#include "fatehov_k_matrix_max_elem/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"

namespace fatehov_k_matrix_max_elem {

FatehovKMatrixMaxElemSEQ::FatehovKMatrixMaxElemSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKMatrixMaxElemSEQ::ValidationImpl() {
  auto &data = GetInput();
  return (std::get<0>(data) > 0 && std::get<0>(data) <= kMaxRows) &&
         (std::get<1>(data) > 0 && std::get<1>(data) <= kMaxCols) &&
         (std::get<0>(data) * std::get<1>(data) <= kMaxMatrixSize) &&
         (std::get<2>(data).size() <= kMaxMatrixSize &&
          std::get<2>(data).size() == std::get<0>(data) * std::get<1>(data)) &&
         (!std::get<2>(data).empty());
}

bool FatehovKMatrixMaxElemSEQ::PreProcessingImpl() {
  return true;
}

bool FatehovKMatrixMaxElemSEQ::RunImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t columns = std::get<1>(data);
  std::vector<double> &matrix = std::get<2>(data);
  double max = matrix[0];
  for (size_t i = 0; i < rows * columns; i++) {
    max = std::max(matrix[i], max);
  }
  GetOutput() = max;
  return true;
}

bool FatehovKMatrixMaxElemSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_max_elem
