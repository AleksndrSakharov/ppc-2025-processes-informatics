#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"
#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const std::size_t kMatrixSize_ = 5000;
  InType input_data_;

  void SetUp() override {
    input_data_.resize(kMatrixSize_, std::vector<int>(kMatrixSize_));

    int counter = 1;
    for (std::size_t ii = 0; ii < kMatrixSize_; ++ii) {
      for (std::size_t jj = 0; jj < kMatrixSize_; ++jj) {
        input_data_[ii][jj] = counter++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = GetTestInputData();

    if (matrix.empty() || output_data.empty()) {
      return false;
    }

    if (output_data.size() != matrix[0].size()) {
      return false;
    }

    std::vector<int> expected(matrix[0].size(), std::numeric_limits<int>::min());
    for (size_t jj = 0; jj < matrix[0].size(); ++jj) {
      for (size_t ii = 0; ii < matrix.size(); ++ii) {
        expected[jj] = std::max(matrix[ii][jj], expected[jj]);
      }
    }

    for (size_t ii = 0; ii < output_data.size(); ++ii) {
      if (output_data[ii] != expected[ii]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(EgorovaLRunPerfTestProcesses, EgorovaLRunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EgorovaLFindMaxValColMatrixMPI, EgorovaLFindMaxValColMatrixSEQ>(
        PPC_SETTINGS_egorova_l_find_max_val_col_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EgorovaLRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(EgorovaLRunModeTests, EgorovaLRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace egorova_l_find_max_val_col_matrix
