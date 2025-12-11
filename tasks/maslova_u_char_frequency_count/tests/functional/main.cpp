#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "maslova_u_char_frequency_count/common/include/common.hpp"
#include "maslova_u_char_frequency_count/mpi/include/ops_mpi.hpp"
#include "maslova_u_char_frequency_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace maslova_u_char_frequency_count {

class MaslovaURunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  MaslovaURunFuncTestsProcesses() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_ = 0;
};

namespace {

TEST_P(MaslovaURunFuncTestsProcesses, charFrequencyCount) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kTestParam = {
    {std::make_tuple(InType("", 'a'), 0, "empty_string"), std::make_tuple(InType("a", 'a'), 1, "single_char_match"),
     std::make_tuple(InType("b", 'x'), 0, "single_char_no_match"),
     std::make_tuple(InType("maslova", 'a'), 2, "normal_string_az"),
     std::make_tuple(InType("zzzzzzzz", 'u'), 0, "all_other_chars"),
     std::make_tuple(InType("aaaaaaaa", 'a'), 8, "all_input_chars"),
     std::make_tuple(InType("Hello World!", 'l'), 3, "mixed_symbols"),
     std::make_tuple(InType("1234567890", '1'), 1, "string_with_numbers"),
     std::make_tuple(InType("123456789abcdlfn", 'a'), 1, "aligned_string_16"),
     std::make_tuple(InType("123456789abcd", 'a'), 1, "unaligned_string_13"),
     std::make_tuple(InType("+++--!", '+'), 3, "string_with_oper"),
     std::make_tuple(InType("1234567890sexdcrfvgbhjnkml", '1'), 1, "string_with_lettes_and_numbers"),
     std::make_tuple(InType(std::string(100, 'x') + std::string(50, 'a'), 'x'), 100, "long_string")}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<MaslovaUCharFrequencyCountMPI, InType>(
                                               kTestParam, PPC_SETTINGS_maslova_u_char_frequency_count),
                                           ppc::util::AddFuncTask<MaslovaUCharFrequencyCountSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_maslova_u_char_frequency_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kTestName = MaslovaURunFuncTestsProcesses::PrintFuncTestName<MaslovaURunFuncTestsProcesses>;
INSTANTIATE_TEST_SUITE_P(charFreqTests, MaslovaURunFuncTestsProcesses, kGtestValues, kTestName);

}  // namespace
}  // namespace maslova_u_char_frequency_count
