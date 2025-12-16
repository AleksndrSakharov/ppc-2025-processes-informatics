#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "kruglova_a_max_diff_adjacent/mpi/include/ops_mpi.hpp"
#include "kruglova_a_max_diff_adjacent/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_max_diff_adjacent {

class KruglovaAMaxDiffAdjacentPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kCount = 4000000;
  InType input_data;
  OutType expected_output = 0.0F;

  void SetUp() override {
    input_data.resize(kCount);

    float acc = 0.0F;
    for (int i = 0; i < kCount; ++i) {
      float step = (static_cast<float>((i % 5) - 2) * 0.7F) + (static_cast<float>((i % 3) - 1) * 0.3F);
      acc += step;
      input_data[i] = acc;
    }

    if (input_data.size() > 1) {
      for (size_t i = 0; i + 1 < input_data.size(); ++i) {
        float diff = std::abs(input_data[i + 1] - input_data[i]);
        expected_output = std::max(diff, expected_output);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(KruglovaAMaxDiffAdjacentPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KruglovaAMaxDiffAdjacentMPI, KruglovaAMaxDiffAdjacentSEQ>(
        PPC_SETTINGS_kruglova_a_max_diff_adjacent);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KruglovaAMaxDiffAdjacentPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KruglovaAMaxDiffAdjacentPerfTests, kGtestValues, kPerfTestName);

}  // namespace kruglova_a_max_diff_adjacent
