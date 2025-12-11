#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_min_val_rows_matrix {

class LevonychevIMinValRowsMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int ROWS_ = 1024;
  const int COLS_ = 262144;
  InType input_data_;
  OutType expected_result_;
  void SetUp() override {
    std::vector<int> matrix(static_cast<size_t>(ROWS_) * static_cast<size_t>(COLS_));
    for (int i = 0; i < COLS_ * ROWS_; ++i) {
      matrix[i] = i;
    }
    input_data_ = std::make_tuple(std::move(matrix), ROWS_, COLS_);
    expected_result_.reserve(ROWS_);
    for (int i = 0; i < ROWS_ * COLS_; i += COLS_) {
      expected_result_.push_back(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LevonychevIMinValRowsMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMinValRowsMatrixMPI, LevonychevIMinValRowsMatrixSEQ>(
        PPC_SETTINGS_levonychev_i_min_val_rows_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMinValRowsMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMinValRowsMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_min_val_rows_matrix
