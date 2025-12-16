#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"
#include "gasenin_l_lex_dif/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_lex_dif {

class GaseninLRunFuncTestsLexDif : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_name = std::get<1>(params);

    if (test_name == "apple_banana") {
      input_data_ = {"apple", "banana"};
      expected_output_ = -1;
    } else if (test_name == "hello_hello") {
      input_data_ = {"hello", "hello"};
      expected_output_ = 0;
    } else if (test_name == "zebra_apple") {
      input_data_ = {"zebra", "apple"};
      expected_output_ = 1;
    } else if (test_name == "empty_first") {
      input_data_ = {"", "test"};
      expected_output_ = -1;
    } else if (test_name == "empty_second") {
      input_data_ = {"test", ""};
      expected_output_ = 1;
    } else if (test_name == "both_empty") {
      input_data_ = {"", ""};
      expected_output_ = 0;
    } else if (test_name == "different_length") {
      input_data_ = {"abc", "abcd"};
      expected_output_ = -1;
    } else if (test_name == "same_prefix") {
      input_data_ = {"abcdef", "abcxyz"};
      expected_output_ = -1;
    } else if (test_name == "unicode_test") {
      input_data_ = {"привет", "пока"};
      expected_output_ = 1;
    } else if (test_name == "case_sensitive") {
      input_data_ = {"Apple", "apple"};
      expected_output_ = -1;
    } else if (test_name == "max_len_equal") {
      std::string long_str(10000, 'a');
      input_data_ = {long_str, long_str};
      expected_output_ = 0;
    } else if (test_name == "max_len_diff_end") {
      std::string str1 = std::string(9999, 'a') + 'b';
      std::string str2 = std::string(9999, 'a') + 'c';
      input_data_ = {str1, str2};
      expected_output_ = -1;
    } else {
      input_data_ = {"test", "test"};
      expected_output_ = 0;
    }
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

TEST_P(GaseninLRunFuncTestsLexDif, LexicographicComparison) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(1, "apple_banana"),     std::make_tuple(2, "hello_hello"),
    std::make_tuple(3, "zebra_apple"),      std::make_tuple(4, "empty_first"),
    std::make_tuple(5, "empty_second"),     std::make_tuple(6, "both_empty"),
    std::make_tuple(7, "different_length"), std::make_tuple(8, "same_prefix"),
    std::make_tuple(9, "unicode_test"),     std::make_tuple(10, "case_sensitive"),
    std::make_tuple(11, "max_len_equal"),   std::make_tuple(12, "max_len_diff_end")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GaseninLLexDifMPI, InType>(kTestParam, PPC_SETTINGS_gasenin_l_lex_dif),
                   ppc::util::AddFuncTask<GaseninLLexDifSEQ, InType>(kTestParam, PPC_SETTINGS_gasenin_l_lex_dif));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaseninLRunFuncTestsLexDif::PrintFuncTestName<GaseninLRunFuncTestsLexDif>;

INSTANTIATE_TEST_SUITE_P(LexDifTests, GaseninLRunFuncTestsLexDif, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_lex_dif
