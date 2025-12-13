#include "smyshlaev_a_mat_mul/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"

namespace smyshlaev_a_mat_mul {

SmyshlaevAMatMulSEQ::SmyshlaevAMatMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SmyshlaevAMatMulSEQ::ValidationImpl() {
  const auto &num_rows_a = std::get<0>(GetInput());
  const auto &mat_a = std::get<1>(GetInput());
  const auto &num_rows_b = std::get<2>(GetInput());
  const auto &mat_b = std::get<3>(GetInput());

  if (num_rows_a <= 0 || num_rows_b <= 0) {
    return false;
  }

  if (mat_a.empty() || mat_b.empty()) {
    return false;
  }

  if (mat_a.size() % num_rows_a != 0 || mat_b.size() % num_rows_b != 0) {
    return false;
  }

  const auto &num_cols_a = static_cast<int>(mat_a.size()) / num_rows_a;
  return (num_cols_a == num_rows_b);
}

bool SmyshlaevAMatMulSEQ::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAMatMulSEQ::RunImpl() {
  const auto &num_rows_a = std::get<0>(GetInput());
  const auto &mat_a = std::get<1>(GetInput());
  const auto &num_rows_b = std::get<2>(GetInput());
  const auto &mat_b = std::get<3>(GetInput());

  const auto &num_cols_b = static_cast<int>(mat_b.size()) / num_rows_b;
  const auto &num_cols_a = num_rows_b;

  std::vector<double> result(static_cast<size_t>(num_rows_a) * num_cols_b, 0.0);

  for (int i = 0; i < num_rows_a; ++i) {
    for (int j = 0; j < num_cols_b; ++j) {
      double sum = 0.0;
      for (int k = 0; k < num_cols_a; ++k) {
        sum += mat_a[(i * num_cols_a) + k] * mat_b[(k * num_cols_b) + j];
      }
      result[(i * num_cols_b) + j] = sum;
    }
  }

  GetOutput() = result;
  return true;
}

bool SmyshlaevAMatMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_mat_mul
