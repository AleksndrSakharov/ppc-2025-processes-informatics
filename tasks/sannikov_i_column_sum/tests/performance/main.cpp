#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "sannikov_i_column_sum/mpi/include/ops_mpi.hpp"
#include "sannikov_i_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
namespace sannikov_i_column_sum {

class SannikovIColumnSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = InType(10000, std::vector<int>(10000));
    for (int i = 0; i < 10000; i++) {
      for (int j = 0; j < 10000; j++) {
        input_data_[i][j] = (i * 14) + (j * 21);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    OutType sums_vec(input_data_.front().size(), 0);
    for (std::size_t i = 0; i < input_data_.size(); i++) {
      for (std::size_t j = 0; j < input_data_[i].size(); j++) {
        sums_vec[j] += input_data_[i][j];
      }
    }
    return sums_vec == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SannikovIColumnSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SannikovIColumnSumMPI, SannikovIColumnSumSEQ>(
    PPC_SETTINGS_sannikov_i_column_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SannikovIColumnSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SannikovIColumnSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace sannikov_i_column_sum
