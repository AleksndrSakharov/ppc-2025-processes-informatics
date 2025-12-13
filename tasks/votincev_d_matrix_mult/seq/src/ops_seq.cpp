#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

VotincevDMatrixMultSEQ::VotincevDMatrixMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VotincevDMatrixMultSEQ::ValidationImpl() {
  const auto &in = GetInput();
  int param_m = std::get<0>(in);
  int param_n = std::get<1>(in);
  int param_k = std::get<2>(in);
  const auto &matrix_a = std::get<3>(in);
  const auto &matrix_b = std::get<4>(in);

  return (param_m > 0 && param_n > 0 && param_k > 0 && static_cast<int>(matrix_a.size()) == (param_m * param_k) &&
          static_cast<int>(matrix_b.size()) == (param_k * param_n));
}

bool VotincevDMatrixMultSEQ::PreProcessingImpl() {
  return true;
}

bool VotincevDMatrixMultSEQ::RunImpl() {
  int param_m = 0;
  int param_n = 0;
  int param_k = 0;
  std::vector<double> matrix_a;
  std::vector<double> matrix_b;
  std::vector<double> matrix_res;

  auto &in = GetInput();
  param_m = std::get<0>(in);
  param_n = std::get<1>(in);
  param_k = std::get<2>(in);
  matrix_a = std::get<3>(in);
  matrix_b = std::get<4>(in);

  auto res_sz = static_cast<size_t>(param_m) * static_cast<size_t>(param_n);
  matrix_res.assign(res_sz, 0.0);

  for (int i = 0; i < param_m; ++i) {
    for (int j = 0; j < param_n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < param_k; ++k) {
        sum += matrix_a[(i * param_k) + k] * matrix_b[(k * param_n) + j];
      }
      matrix_res[(i * param_n) + j] = sum;
    }
  }

  GetOutput() = matrix_res;
  return true;
}

bool VotincevDMatrixMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_matrix_mult
