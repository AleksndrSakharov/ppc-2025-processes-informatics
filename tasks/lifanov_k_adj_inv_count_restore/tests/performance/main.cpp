#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "lifanov_k_adj_inv_count_restore/common/include/common.hpp"
#include "lifanov_k_adj_inv_count_restore/mpi/include/ops_mpi.hpp"
#include "lifanov_k_adj_inv_count_restore/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lifanov_k_adj_inv_count_restore {

class LifanovKRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kSize = 100'000'000;

 protected:
  void SetUp() override {
    input_data_.resize(kSize);

    input_data_[0] = 100;
    for (std::size_t i = 1; i + 1 < kSize; ++i) {
      input_data_[i] = static_cast<int>(i);
    }

    input_data_[kSize - 1] = 0;
    expected_ = 2;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

 private:
  InType input_data_;
  OutType expected_{};
};

TEST_P(LifanovKRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

auto BuildPerfParams() {
  const auto all_tasks =
      ppc::util::MakeAllPerfTasks<InType, LifanovKAdjacentInversionCountMPI, LifanovKAdjacentInversionCountSEQ>(
          PPC_SETTINGS_lifanov_k_adj_inv_count_restore);

  return ppc::util::TupleToGTestValues(all_tasks);
}

const auto kPerfParams = BuildPerfParams();

std::string PerfName(const ::testing::TestParamInfo<LifanovKRunPerfTests::ParamType> &info) {
  return LifanovKRunPerfTests::CustomPerfTestName(info);
}

INSTANTIATE_TEST_SUITE_P(RunModeTests, LifanovKRunPerfTests, kPerfParams, PerfName);

}  // namespace
}  // namespace lifanov_k_adj_inv_count_restore
