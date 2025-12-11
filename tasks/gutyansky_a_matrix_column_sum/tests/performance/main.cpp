#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"
#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_matrix_column_sum {

class GutyanskyAMatrixColumnSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    if (ShouldLoadDataAndTest()) {
      LoadTestData();
      return;
    }

    InitializeEmptyData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ShouldLoadDataAndTest()) {
      return output_data_ == output_data;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const size_t kSize_ = 16000;
  InType input_data_ = {};
  OutType output_data_;

  [[nodiscard]] static bool ShouldLoadDataAndTest() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    if (test_name.find("_mpi") == std::string::npos) {
      return true;
    }

    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }

  void InitializeEmptyData() {
    input_data_ = {.rows = 0, .cols = 0, .data = {}};
    output_data_.clear();
  }

  void LoadTestData() {
    input_data_.rows = kSize_;
    input_data_.cols = kSize_;
    input_data_.data.assign(kSize_ * kSize_, 1);
    output_data_.assign(kSize_, static_cast<int32_t>(kSize_));
  }
};

TEST_P(GutyanskyAMatrixColumnSumPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GutyanskyAMatrixColumnSumMPI, GutyanskyAMatrixColumnSumSEQ>(
        PPC_SETTINGS_gutyansky_a_matrix_column_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GutyanskyAMatrixColumnSumPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GutyanskyAMatrixColumnSumPerfTest, kGtestValues, kPerfTestName);

}  // namespace gutyansky_a_matrix_column_sum
