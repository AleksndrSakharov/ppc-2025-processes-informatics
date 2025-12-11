#include <gtest/gtest.h>

#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "baldin_a_word_count/common/include/common.hpp"
#include "baldin_a_word_count/mpi/include/ops_mpi.hpp"
#include "baldin_a_word_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_word_count {

namespace {

bool IsWordChar(char c) {
  return ((std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '-' || c == '_');
}

}  // namespace

class BaldinAWordCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_baldin_a_word_count, "book-war-and-peace.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("File reading error");
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    input_data_ = buffer.str();
    file.close();

    size_t count = 0;
    bool in_word = false;
    for (char c : input_data_) {
      if (IsWordChar(c)) {
        if (!in_word) {
          in_word = true;
          count++;
        }
      } else {
        in_word = false;
      }
    }
    expected_output_ = count;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  BaldinAWordCountPerfTests() = default;
};

TEST_P(BaldinAWordCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BaldinAWordCountMPI, BaldinAWordCountSEQ>(PPC_SETTINGS_baldin_a_word_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaldinAWordCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaldinAWordCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace baldin_a_word_count
