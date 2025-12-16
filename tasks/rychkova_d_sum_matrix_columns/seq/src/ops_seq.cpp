#include "rychkova_d_sum_matrix_columns/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "rychkova_d_sum_matrix_columns/common/include/common.hpp"

namespace rychkova_d_sum_matrix_columns {

RychkovaDSumMatrixColumnsSEQ::RychkovaDSumMatrixColumnsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput()[i] = in[i];
  }
  GetOutput() = OutType{};
}

bool RychkovaDSumMatrixColumnsSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    return true;
  }

  size_t cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool RychkovaDSumMatrixColumnsSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    GetOutput().clear();
    return true;
  }

  const size_t cols = input[0].size();
  GetOutput().assign(cols, 0);
  return true;
}

bool RychkovaDSumMatrixColumnsSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  if (input.empty()) {
    return true;
  }

  if (output.empty() || output.size() != input[0].size()) {
    output.assign(input[0].size(), 0);
  }

  for (const auto &row : input) {
    for (size_t j = 0; j < row.size(); ++j) {
      output[j] += row[j];
    }
  }

  return true;
}

bool RychkovaDSumMatrixColumnsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rychkova_d_sum_matrix_columns
