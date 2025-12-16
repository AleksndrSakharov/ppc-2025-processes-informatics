#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "papulina_y_simple_iteration/common/include/common.hpp"
#include "papulina_y_simple_iteration/mpi/include/ops_mpi.hpp"
#include "papulina_y_simple_iteration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace papulina_y_simple_iteration {

class PapulinaYSimpleIterationRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expectedResult_;
  double eps_ = 1e-7;

  void SetUp() override {
    size_t n = 200;
    std::vector<double> matrix_a(n * n, 0.0);
    std::vector<double> b(n, 0.0);
    std::vector<double> expected_x(n, 1.0);

    for (size_t i = 0; i < n; i++) {
      matrix_a[(i * n) + i] = 5.0;

      if (i > 0) {
        matrix_a[(i * n) + (i - 1)] = 2.0;
      }
      if (i < n - 1) {
        matrix_a[(i * n) + (i + 1)] = 2.0;
      }

      if (i > 1) {
        matrix_a[(i * n) + (i - 2)] = 0.5;
      }
      if (i < n - 2) {
        matrix_a[(i * n) + (i + 2)] = 0.5;
      }

      b[i] = matrix_a[(i * n) + i] * 1.0;
      if (i > 0) {
        b[i] += matrix_a[(i * n) + (i - 1)] * 1.0;
      }
      if (i < n - 1) {
        b[i] += matrix_a[(i * n) + (i + 1)] * 1.0;
      }
      if (i > 1) {
        b[i] += matrix_a[(i * n) + (i - 2)] * 1.0;
      }
      if (i < n - 2) {
        b[i] += matrix_a[(i * n) + (i + 2)] * 1.0;
      }
    }

    input_data_ = std::make_tuple(n, matrix_a, b);
    expectedResult_ = expected_x;
  }
  bool CheckTestOutputData(OutType &output_data) final {
    if (expectedResult_.size() != output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); i++) {
      if (std::abs(expectedResult_[i] - output_data[i]) > eps_) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PapulinaYSimpleIterationRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PapulinaYSimpleIterationMPI, PapulinaYSimpleIterationSEQ>(
        PPC_SETTINGS_papulina_y_simple_iteration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PapulinaYSimpleIterationRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PapulinaYSimpleIterationRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace papulina_y_simple_iteration
