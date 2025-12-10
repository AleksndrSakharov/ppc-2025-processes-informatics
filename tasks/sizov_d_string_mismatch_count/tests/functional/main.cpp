#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"
#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_string_mismatch_count {

class SizovDRunFuncTestsStringMismatchCount : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_string_mismatch_count, param + ".txt");

    std::ifstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + file_path);
    }

    std::string a;
    std::string b;
    std::string expected_line;
    std::getline(file, a);
    std::getline(file, b);
    std::getline(file, expected_line);
    file.close();

    TrimString(a);
    TrimString(b);
    TrimString(expected_line);

    if (a.empty() || b.empty()) {
      throw std::runtime_error("Input strings cannot be empty in " + file_path);
    }
    if (a.size() != b.size()) {
      throw std::runtime_error("Input strings must have equal length in " + file_path);
    }

    try {
      expected_result_ = std::stoi(expected_line);
    } catch (...) {
      throw std::runtime_error("Invalid expected value in " + file_path + " (line 3 must be an integer)");
    }

    input_data_ = std::make_tuple(a, b);
    is_valid_ = true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (!is_valid_) {
      return true;
    }
    return output_data == expected_result_;
  }

 private:
  static void TrimString(std::string &s) {
    const auto is_edge_escape = [](unsigned char c) { return c == '\r' || c == '\n' || c == '\t'; };

    std::size_t left = 0;
    while (left < s.size() && is_edge_escape(static_cast<unsigned char>(s[left]))) {
      ++left;
    }

    std::size_t right = s.size();
    while (right > left && is_edge_escape(static_cast<unsigned char>(s[right - 1]))) {
      --right;
    }

    if (left == 0 && right == s.size()) {
      return;
    }

    s.erase(right);
    s.erase(0, left);
  }

  InType input_data_ = std::make_tuple(std::string{}, std::string{});
  OutType expected_result_ = 0;
  bool is_valid_ = true;
};

namespace {

TEST_P(SizovDRunFuncTestsStringMismatchCount, CompareStringsFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"strings1", "strings2", "strings3", "strings4", "strings5",
                                             "strings6", "strings7", "strings8", "strings9", "strings10"};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sizov_d_string_mismatch_count::SizovDStringMismatchCountMPI, InType>(
                       kTestParam, PPC_SETTINGS_sizov_d_string_mismatch_count),
                   ppc::util::AddFuncTask<sizov_d_string_mismatch_count::SizovDStringMismatchCountSEQ, InType>(
                       kTestParam, PPC_SETTINGS_sizov_d_string_mismatch_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = SizovDRunFuncTestsStringMismatchCount::PrintFuncTestName<SizovDRunFuncTestsStringMismatchCount>;

INSTANTIATE_TEST_SUITE_P(SizovDStringMismatchCount, SizovDRunFuncTestsStringMismatchCount, kGtestValues, kTestName);

}  // namespace
}  // namespace sizov_d_string_mismatch_count
