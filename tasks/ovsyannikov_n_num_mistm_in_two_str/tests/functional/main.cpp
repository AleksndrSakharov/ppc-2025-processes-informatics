#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>
#include <utility>

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

class OvsyannikovNRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<TestType>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &actual_res) final {
    return actual_res == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_ = 0;
};

TEST_P(OvsyannikovNRunFuncTestsProcesses, CalculateMismatches) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    {std::make_tuple(std::make_pair("", ""), 0, "EmptyStrings"),
     std::make_tuple(std::make_pair("a", "a"), 0, "SingleCharMatch"),
     std::make_tuple(std::make_pair("a", "b"), 1, "SingleCharMismatch"),
     std::make_tuple(std::make_pair("hello", "hello"), 0, "IdenticalStrings"),
     std::make_tuple(std::make_pair("abc", "def"), 3, "AllMismatch"),
     std::make_tuple(std::make_pair("apple", "apply"), 1, "LastCharMismatch"),
     std::make_tuple(std::make_pair("Case", "case"), 1, "CaseSensitivity"),
     std::make_tuple(std::make_pair("12345", "12355"), 1, "NumbersMismatch"),
     std::make_tuple(std::make_pair("two words", "two_words"), 1, "SpaceVsUnderscore"),
     std::make_tuple(std::make_pair("longstringtest", "longstringbest"), 1, "LongStringOneMismatch")}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<OvsyannikovNNumMistmInTwoStrMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ovsyannikov_n_num_mistm_in_two_str),
                                           ppc::util::AddFuncTask<OvsyannikovNNumMistmInTwoStrSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ovsyannikov_n_num_mistm_in_two_str));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = OvsyannikovNRunFuncTestsProcesses::PrintFuncTestName<OvsyannikovNRunFuncTestsProcesses>;
INSTANTIATE_TEST_SUITE_P(NumMistmInTwoStr, OvsyannikovNRunFuncTestsProcesses, kGtestValues, kPerfTestName);
}  // namespace ovsyannikov_n_num_mistm_in_two_str
