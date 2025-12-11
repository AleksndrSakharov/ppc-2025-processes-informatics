#include "gutyansky_a_matrix_column_sum/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"

namespace gutyansky_a_matrix_column_sum {

GutyanskyAMatrixColumnSumSEQ::GutyanskyAMatrixColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMatrixColumnSumSEQ::ValidationImpl() {
  return GetInput().IsValid();
}

bool GutyanskyAMatrixColumnSumSEQ::PreProcessingImpl() {
  return true;
}

bool GutyanskyAMatrixColumnSumSEQ::RunImpl() {
  GetOutput().resize(GetInput().cols);
  std::fill(GetOutput().begin(), GetOutput().end(), 0);

  for (size_t i = 0; i < GetInput().rows; i++) {
    for (size_t j = 0; j < GetInput().cols; j++) {
      GetOutput()[j] += GetInput().data[(i * GetInput().cols) + j];
    }
  }

  return true;
}

bool GutyanskyAMatrixColumnSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_column_sum
