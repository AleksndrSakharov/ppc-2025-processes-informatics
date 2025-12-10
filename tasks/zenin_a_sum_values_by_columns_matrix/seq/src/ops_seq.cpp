#include "zenin_a_sum_values_by_columns_matrix/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

ZeninASumValuesByColumnsMatrixSEQ::ZeninASumValuesByColumnsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninASumValuesByColumnsMatrixSEQ::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) * std::get<1>(input)) == std::get<2>(input).size() && (GetOutput().empty());
}

bool ZeninASumValuesByColumnsMatrixSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  GetOutput().clear();
  GetOutput().resize(std::get<1>(input), 0.0);
  return true;
}

bool ZeninASumValuesByColumnsMatrixSEQ::RunImpl() {
  auto &input = GetInput();
  auto &rows = std::get<0>(input);
  auto &columns = std::get<1>(input);
  auto &matrix = std::get<2>(input);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < columns; ++col) {
      GetOutput()[col] += matrix[(row * columns) + col];
    }
  }
  return true;
}

bool ZeninASumValuesByColumnsMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_sum_values_by_columns_matrix
