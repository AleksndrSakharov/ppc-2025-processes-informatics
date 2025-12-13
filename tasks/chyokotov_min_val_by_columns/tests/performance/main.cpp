#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"
#include "chyokotov_min_val_by_columns/mpi/include/ops_mpi.hpp"
#include "chyokotov_min_val_by_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chyokotov_min_val_by_columns {

class ChyokotovMinValPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const int cols = 6000;
    const int rows = 6000;

    input_data_.resize(rows);
    for (int i = 0; i < rows; i++) {
      input_data_[i].resize(cols);
      for (int j = 0; j < cols; j++) {
        input_data_[i][j] = i + j;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = input_data_;

    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != matrix[0].size()) {
      return false;
    }

    std::vector<int> expected_output(matrix[0].size(), INT_MAX);

    for (size_t i = 0; i < matrix[0].size(); i++) {
      for (size_t j = 0; j < matrix.size(); j++) {
        expected_output[i] = std::min(expected_output[i], matrix[j][i]);
      }
    }

    return output_data == expected_output;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChyokotovMinValPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChyokotovMinValByColumnsMPI, ChyokotovMinValByColumnsSEQ>(
        PPC_SETTINGS_chyokotov_min_val_by_columns);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChyokotovMinValPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChyokotovMinValPerfTest, kGtestValues, kPerfTestName);

}  // namespace chyokotov_min_val_by_columns
