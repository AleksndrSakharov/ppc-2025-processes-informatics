#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "smyshlaev_a_mat_mul/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_mat_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smyshlaev_a_mat_mul {
const int kRows = 500;
const int kCols = 500;
const int kCommon = 500;

class SmyshlaevAMatMulRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;

  void SetUp() override {
    std::vector<double> a(static_cast<size_t>(kRows) * kCommon);
    std::vector<double> b(static_cast<size_t>(kCommon) * kCols);

    for (int i = 0; i < kRows * kCommon; ++i) {
      a[i] = (i % 5) + 1.0;
    }
    for (int i = 0; i < kCommon * kCols; ++i) {
      b[i] = (i % 5) + 1.0;
    }

    input_data_ = std::make_tuple(kRows, a, kCommon, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int m = std::get<0>(input_data_);
    const auto &mat_a = std::get<1>(input_data_);
    int k = std::get<2>(input_data_);
    const auto &mat_b = std::get<3>(input_data_);
    int n = static_cast<int>(mat_b.size()) / k;

    if (output_data.size() != static_cast<size_t>(m) * n) {
      return false;
    }

    std::vector<double> expected(static_cast<size_t>(m) * n, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int idx = 0; idx < k; ++idx) {
          expected[(i * n) + j] += mat_a[(i * k) + idx] * mat_b[(idx * n) + j];
        }
      }
    }

    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::abs(output_data[i] - expected[i]) > 1e-5) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SmyshlaevAMatMulRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SmyshlaevAMatMulMPI, SmyshlaevAMatMulSEQ>(PPC_SETTINGS_smyshlaev_a_mat_mul);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmyshlaevAMatMulRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmyshlaevAMatMulRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace smyshlaev_a_mat_mul
