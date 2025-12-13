#include "levonychev_i_mult_matrix_vec/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"

namespace levonychev_i_mult_matrix_vec {

LevonychevIMultMatrixVecSEQ::LevonychevIMultMatrixVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LevonychevIMultMatrixVecSEQ::ValidationImpl() {
  const size_t matrix_size = std::get<0>(GetInput()).size();
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  bool is_correct_matrix_size = (matrix_size == static_cast<size_t>(rows) * static_cast<size_t>(cols));
  bool is_correct_vector_size = (static_cast<size_t>(cols) == std::get<3>(GetInput()).size());
  return matrix_size != 0 && rows != 0 && cols != 0 && is_correct_matrix_size && is_correct_vector_size;
}

bool LevonychevIMultMatrixVecSEQ::PreProcessingImpl() {
  GetOutput().resize(static_cast<size_t>(std::get<1>(GetInput())));
  return true;
}

bool LevonychevIMultMatrixVecSEQ::RunImpl() {
  const std::vector<double> &matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  const std::vector<double> &vec_x = std::get<3>(GetInput());

  OutType &result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    double scalar_product = 0;
    for (int j = 0; j < cols; ++j) {
      scalar_product += matrix[(i * cols) + j] * vec_x[j];
    }
    result[i] = scalar_product;
  }
  return true;
}

bool LevonychevIMultMatrixVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_mult_matrix_vec
