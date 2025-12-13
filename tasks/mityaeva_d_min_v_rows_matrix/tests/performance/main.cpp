#include <gtest/gtest.h>

#include <cstddef>

#include "mityaeva_d_min_v_rows_matrix/common/include/common.hpp"
#include "mityaeva_d_min_v_rows_matrix/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_min_v_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace mityaeva_d_min_v_rows_matrix {

class MinValuesInRowsRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 5000;
  InType input_data_;

  void SetUp() override {
    int rows = kMatrixSize_;
    int cols = kMatrixSize_;

    input_data_.clear();
    input_data_.reserve(2 + (rows * cols));

    input_data_.push_back(rows);
    input_data_.push_back(cols);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        input_data_.push_back(((i + j) % 1000) + 1);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    int result_rows = output_data[0];
    if (result_rows <= 0) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); ++i) {
      if (output_data[i] < 1 || output_data[i] > 1000) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MinValuesInRowsRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MinValuesInRowsMPI, MinValuesInRowsSEQ>(
    PPC_SETTINGS_mityaeva_d_min_v_rows_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MinValuesInRowsRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MinValuesInRowsRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace mityaeva_d_min_v_rows_matrix
