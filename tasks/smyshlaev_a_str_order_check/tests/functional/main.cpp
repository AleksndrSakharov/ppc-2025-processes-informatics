#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"
#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_str_order_check/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_str_order_check {

class SmyshlaevAStrOrderCheckRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::make_pair(std::get<0>(params), std::get<1>(params));
    expected_output_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(SmyshlaevAStrOrderCheckRunFuncTestsProcesses, StringOrderCheckTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(std::string("car"), std::string("car"), 0, std::string("equal_not_empty_strings")),
    std::make_tuple(std::string("air"), std::string("bike"), -1, std::string("first_string_less_than_second")),
    std::make_tuple(std::string("plane"), std::string("material"), 1, std::string("first_string_greater_than_second")),
    std::make_tuple(std::string("abc"), std::string("abcdefg"), -1, std::string("first_string_prefix_of_second")),
    std::make_tuple(std::string("gdefg"), std::string("gde"), 1, std::string("second_string_prefix_of_first")),
    std::make_tuple(std::string("Abc"), std::string("abc"), -1, std::string("case_sensitive_comparison")),
    std::make_tuple(std::string(""), std::string(""), 0, std::string("both_strings_empty")),
    std::make_tuple(std::string("a"), std::string(""), 1, std::string("right_string_empty")),
    std::make_tuple(std::string(""), std::string("a"), -1, std::string("left_string_empty")),
    std::make_tuple(std::string("a"), std::string("a"), 0, std::string("characters_equal")),
    std::make_tuple(std::string("a"), std::string("b"), -1, std::string("left_character_less"))};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SmyshlaevAStrOrderCheckMPI, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_str_order_check),
    ppc::util::AddFuncTask<SmyshlaevAStrOrderCheckSEQ, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_str_order_check));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SmyshlaevAStrOrderCheckRunFuncTestsProcesses::PrintFuncTestName<SmyshlaevAStrOrderCheckRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(StringOrderCheckTests, SmyshlaevAStrOrderCheckRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace smyshlaev_a_str_order_check
