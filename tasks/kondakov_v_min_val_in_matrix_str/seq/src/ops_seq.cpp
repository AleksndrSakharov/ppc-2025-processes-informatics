#include "kondakov_v_min_val_in_matrix_str/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"

namespace kondakov_v_min_val_in_matrix_str {

KondakovVMinValMatrixSEQ::KondakovVMinValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp = in;
  GetInput().swap(tmp);
  GetOutput().clear();
  GetOutput().resize(in.size());
}

bool KondakovVMinValMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }

  size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return !row.empty() && row.size() == cols; });
}
bool KondakovVMinValMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool KondakovVMinValMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &output = GetOutput();

  for (size_t i = 0; i < matrix.size(); ++i) {
    int min_val = std::numeric_limits<int>::max();
    for (int val : matrix[i]) {
      min_val = std::min(min_val, val);
    }
    output[i] = min_val;
  }

  return true;
}

bool KondakovVMinValMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_min_val_in_matrix_str
