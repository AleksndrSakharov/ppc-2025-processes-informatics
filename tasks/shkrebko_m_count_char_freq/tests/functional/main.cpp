#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "shkrebko_m_count_char_freq/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_count_char_freq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_count_char_freq {

class ShkrebkoMCountCharFreqFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (test_name == "test1") {
      input_data_ = std::make_tuple("Alolo polo", "l");
      expected_data_ = 3;
    } else if (test_name == "test2") {
      input_data_ = std::make_tuple("aramopma", "m");
      expected_data_ = 2;
    } else if (test_name == "test3") {
      input_data_ = std::make_tuple("banana", "a");
      expected_data_ = 3;
    } else if (test_name == "test4") {
      input_data_ = std::make_tuple("abcde", "z");
      expected_data_ = 0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_{0};
};

namespace {

TEST_P(ShkrebkoMCountCharFreqFuncTests, CountCharFrequency) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {"test1", "test2", "test3", "test4"};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<shkrebko_m_count_char_freq::ShkrebkoMCountCharFreqMPI, InType>(
                       kTestParam, PPC_SETTINGS_shkrebko_m_count_char_freq),
                   ppc::util::AddFuncTask<shkrebko_m_count_char_freq::ShkrebkoMCountCharFreqSEQ, InType>(
                       kTestParam, PPC_SETTINGS_shkrebko_m_count_char_freq));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkrebkoMCountCharFreqFuncTests::PrintFuncTestName<ShkrebkoMCountCharFreqFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShkrebkoMCharFreq, ShkrebkoMCountCharFreqFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shkrebko_m_count_char_freq
