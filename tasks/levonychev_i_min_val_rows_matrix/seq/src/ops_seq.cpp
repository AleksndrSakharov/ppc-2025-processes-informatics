#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixSEQ::LevonychevIMinValRowsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LevonychevIMinValRowsMatrixSEQ::ValidationImpl() {
  const size_t vector_size = std::get<0>(GetInput()).size();
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  return vector_size != 0 && rows != 0 && cols != 0 &&
         (vector_size == static_cast<size_t>(rows) * static_cast<size_t>(cols));
}

bool LevonychevIMinValRowsMatrixSEQ::PreProcessingImpl() {
  GetOutput().resize(static_cast<size_t>(std::get<1>(GetInput())));
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::RunImpl() {
  const std::vector<int> &matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  OutType &result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    int min_val = matrix[static_cast<size_t>(cols) * static_cast<size_t>(i)];
    for (int j = 1; j < cols; ++j) {
      min_val = std::min(matrix[(cols * i) + j], min_val);
    }
    result[i] = min_val;
  }

  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_min_val_rows_matrix
