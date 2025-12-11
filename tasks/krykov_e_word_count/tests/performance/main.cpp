#include <gtest/gtest.h>

#include <string>

#include "krykov_e_word_count/common/include/common.hpp"
#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"
#include "krykov_e_word_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krykov_e_word_count {

class KrykovEWordCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int kRepeatCount_ = 2000000;
  InType input_data_;
  OutType expected_result_ = 0;

  void SetUp() override {
    std::string base_text = "word ";
    input_data_.clear();
    for (int i = 0; i < kRepeatCount_; i++) {
      input_data_ += base_text;
    }

    expected_result_ = kRepeatCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  KrykovEWordCountPerfTests() = default;
};

TEST_P(KrykovEWordCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KrykovEWordCountMPI, KrykovEWordCountSEQ>(PPC_SETTINGS_krykov_e_word_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrykovEWordCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrykovEWordCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace krykov_e_word_count
