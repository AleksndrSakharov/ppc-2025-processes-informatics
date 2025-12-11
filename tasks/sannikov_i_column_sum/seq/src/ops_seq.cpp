#include "sannikov_i_column_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
namespace sannikov_i_column_sum {

SannikovIColumnSumSEQ::SannikovIColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);

  GetOutput().clear();
}

bool SannikovIColumnSumSEQ::ValidationImpl() {
  const auto &input_matrix = GetInput();
  if (input_matrix.empty() || input_matrix.front().empty()) {
    return false;
  }

  const std::size_t columns = input_matrix.front().size();
  for (const auto &row : input_matrix) {
    if (row.size() != columns) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIColumnSumSEQ::PreProcessingImpl() {
  const auto &input_matrix = GetInput();
  GetOutput().clear();
  GetOutput().resize(input_matrix.front().size(), 0);
  return !GetOutput().empty();
}

bool SannikovIColumnSumSEQ::RunImpl() {
  const auto &input_matrix = GetInput();
  for (const auto &row : input_matrix) {
    std::size_t column = 0;
    for (const auto &value : row) {
      GetOutput()[column] += value;
      column++;
    }
  }
  return !GetOutput().empty();
}

bool SannikovIColumnSumSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_column_sum
