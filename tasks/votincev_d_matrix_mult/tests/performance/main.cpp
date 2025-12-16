#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"
#include "votincev_d_matrix_mult/mpi/include/ops_mpi.hpp"
#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  InType GetTestInputData() final {
    return input_data;
  }

 protected:
  InType input_data;
  OutType expected_res;

  void SetUp() override {
    size_t param_m = 0;
    size_t param_n = 0;
    size_t param_k = 0;

    param_m = 600;
    param_n = param_m;
    param_k = param_m;
    std::vector<double> matrix_a(param_m * param_k);
    std::vector<double> matrix_b(param_k * param_n);

    double sgn_swapper = 1;
    for (size_t i = 0; i < param_m * param_k; i++) {
      matrix_a[i] = static_cast<double>(i % 5) * sgn_swapper;
      sgn_swapper *= -1;
    }

    sgn_swapper = -1;
    for (size_t i = 0; i < param_k * param_n; i++) {
      matrix_b[i] = static_cast<double>(i % 5) * sgn_swapper;
      sgn_swapper *= -1;
    }

    input_data = std::make_tuple(param_m, param_n, param_k, matrix_a, matrix_b);

    // вычисляю предполагаемый результат
    expected_res.assign(param_m * param_n, 0.0);
    for (size_t i = 0; i < param_m; ++i) {
      for (size_t j = 0; j < param_n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < param_k; ++k) {
          sum += matrix_a[(i * param_k) + k] * matrix_b[(k * param_n) + j];
        }
        expected_res[(i * param_n) + j] = sum;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 1,2... процессы не владеют нужным результатом
    if (output_data.size() != expected_res.size()) {
      return true;
    }
    // 0й процесс должен иметь корректную матрицу после умножения
    return output_data == expected_res;
  }
};

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VotincevDMatrixMultMPI, VotincevDMatrixMultSEQ>(
    PPC_SETTINGS_votincev_d_matrix_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = VotincevDMatrixMultRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(VotincevDMatrixMultRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, VotincevDMatrixMultRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace votincev_d_matrix_mult
