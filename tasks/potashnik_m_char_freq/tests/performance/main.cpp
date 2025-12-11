#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "potashnik_m_char_freq/common/include/common.hpp"
#include "potashnik_m_char_freq/mpi/include/ops_mpi.hpp"
#include "potashnik_m_char_freq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace potashnik_m_char_freq {

class PotashnikMCharFreqPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100000000;
  InType input_data_;

  void SetUp() override {
    std::string str;
    char chr = 0;

    chr = 'k';
    for (int i = 0; i < kCount_; i++) {
      char c = static_cast<char>('a' + ((i * 7 + 13) % 26));
      str += c;
    }

    input_data_ = std::make_tuple(str, chr);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int res = 0;
    std::string str = std::get<0>(input_data_);
    char chr = std::get<1>(input_data_);

    int string_size = static_cast<int>(str.size());
    for (int i = 0; i < string_size; i++) {
      if (str[i] == chr) {
        res++;
      }
    }

    return (res == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PotashnikMCharFreqPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PotashnikMCharFreqMPI, PotashnikMCharFreqSEQ>(
    PPC_SETTINGS_potashnik_m_char_freq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PotashnikMCharFreqPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PotashnikMCharFreqPerfTests, kGtestValues, kPerfTestName);

}  // namespace potashnik_m_char_freq
