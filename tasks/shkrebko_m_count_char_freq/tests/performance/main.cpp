#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "shkrebko_m_count_char_freq/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_count_char_freq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkrebko_m_count_char_freq {

class ShkrebkoMCountCharFreqPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_data_{0};

  void SetUp() override {
    std::string large_text;
    const int target_size = 100000000;
    const std::string base_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    while (large_text.size() < target_size) {
      large_text += base_text;
    }

    large_text = large_text.substr(0, target_size);

    std::string target_char = "e";

    expected_data_ = 0;
    for (char c : large_text) {
      if (c == target_char[0]) {
        expected_data_++;
      }
    }

    input_data_ = std::make_tuple(large_text, target_char);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShkrebkoMCountCharFreqPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ShkrebkoMCountCharFreqMPI, ShkrebkoMCountCharFreqSEQ>(
    PPC_SETTINGS_shkrebko_m_count_char_freq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrebkoMCountCharFreqPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrebkoMCountCharFreqPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkrebko_m_count_char_freq
