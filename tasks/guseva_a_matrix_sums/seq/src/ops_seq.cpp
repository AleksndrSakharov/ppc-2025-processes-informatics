#include "guseva_a_matrix_sums/seq/include/ops_seq.hpp"

#include <cstdint>

#include "guseva_a_matrix_sums/common/include/common.hpp"

namespace guseva_a_matrix_sums {

GusevaAMatrixSumsSEQ::GusevaAMatrixSumsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GusevaAMatrixSumsSEQ::ValidationImpl() {
  return (static_cast<uint64_t>(std::get<0>(GetInput())) * std::get<1>(GetInput()) == std::get<2>(GetInput()).size()) &&
         (GetOutput().empty());
}

bool GusevaAMatrixSumsSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(std::get<1>(GetInput()), 0.0);
  return true;
}

bool GusevaAMatrixSumsSEQ::RunImpl() {
  const auto &[rows, columns, matrix] = GetInput();
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < columns; j++) {
      GetOutput()[j] += matrix[(i * columns) + j];
    }
  }
  return true;
}

bool GusevaAMatrixSumsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace guseva_a_matrix_sums
