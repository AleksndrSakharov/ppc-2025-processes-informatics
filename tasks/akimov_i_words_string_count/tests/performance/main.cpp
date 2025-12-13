#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "akimov_i_words_string_count/mpi/include/ops_mpi.hpp"
#include "akimov_i_words_string_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akimov_i_words_string_count {

class AkimovIWordsStringCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const std::size_t words = 1'000'000;
    std::string s;
    s.reserve(words * 5);
    for (std::size_t i = 0; i < words; ++i) {
      s += "word";
      if (i + 1 != words) {
        s += ' ';
      }
    }
    input_data_ = InType(s.begin(), s.end());
    expected_result_ = static_cast<int>(words);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
};

TEST_P(AkimovIWordsStringCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, AkimovIWordsStringCountMPI, AkimovIWordsStringCountSEQ>(
    PPC_SETTINGS_akimov_i_words_string_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AkimovIWordsStringCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AkimovIWordsStringCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace akimov_i_words_string_count
