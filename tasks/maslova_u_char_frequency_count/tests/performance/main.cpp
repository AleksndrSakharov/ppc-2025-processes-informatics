#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "maslova_u_char_frequency_count/common/include/common.hpp"
#include "maslova_u_char_frequency_count/mpi/include/ops_mpi.hpp"
#include "maslova_u_char_frequency_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maslova_u_char_frequency_count {

class MaslovaUPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_{};

 protected:
  void SetUp() override {
    const size_t str_size = 100000000;
    const char input_char = 'y';

    std::string generated_string(str_size, 'a');
    size_t expected_count = 0;

    if constexpr (str_size > 100) {
      for (size_t i = 0; i < str_size / 100; ++i) {
        generated_string[i * 50] = input_char;
        expected_count++;
      }
    }

    input_data_ = {generated_string, input_char};
    expected_output_ = expected_count;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }
};

TEST_P(MaslovaUPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MaslovaUCharFrequencyCountMPI, MaslovaUCharFrequencyCountSEQ>(
        PPC_SETTINGS_maslova_u_char_frequency_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaslovaUPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaslovaUPerfTests, kGtestValues, kPerfTestName);

}  // namespace maslova_u_char_frequency_count
