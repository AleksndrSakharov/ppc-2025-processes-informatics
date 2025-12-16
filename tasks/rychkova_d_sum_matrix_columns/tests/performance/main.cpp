#include <gtest/gtest.h>

#include <cstddef>

#include "rychkova_d_sum_matrix_columns/common/include/common.hpp"
#include "rychkova_d_sum_matrix_columns/mpi/include/ops_mpi.hpp"
#include "rychkova_d_sum_matrix_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rychkova_d_sum_matrix_columns {

class RychkovaRunPerfTestMatrixColumns : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kMatrixSize_ = 4000;

  InType input_matrix_;

  void SetUp() override {
    input_matrix_.resize(kMatrixSize_);
    for (size_t i = 0; i < kMatrixSize_; ++i) {
      input_matrix_[i].resize(kMatrixSize_);
      for (size_t j = 0; j < kMatrixSize_; ++j) {
        input_matrix_[i][j] = static_cast<int>(i + j);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != kMatrixSize_) {
      return false;
    }

    for (size_t j = 0; j < kMatrixSize_; ++j) {
      int expected_sum = 0;
      for (size_t i = 0; i < kMatrixSize_; ++i) {
        expected_sum += static_cast<int>(i + j);
      }

      if (output_data[j] != expected_sum) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_matrix_;
  }
};

TEST_P(RychkovaRunPerfTestMatrixColumns, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RychkovaDSumMatrixColumnsMPI, RychkovaDSumMatrixColumnsSEQ>(
        PPC_SETTINGS_rychkova_d_sum_matrix_columns);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RychkovaRunPerfTestMatrixColumns::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RychkovaRunPerfTestMatrixColumns, kGtestValues, kPerfTestName);

}  // namespace rychkova_d_sum_matrix_columns
