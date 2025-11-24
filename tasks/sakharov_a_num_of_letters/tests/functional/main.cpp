#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sakharov_a_num_of_letters/common/include/common.hpp"
#include "sakharov_a_num_of_letters/mpi/include/ops_mpi.hpp"
#include "sakharov_a_num_of_letters/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sakharov_a_num_of_letters {

class SakharovARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    InType in = std::get<0>(params);
    input_data_ = in;

    expected_output_ = 0;
    const std::string &str = std::get<1>(in);
    for (char c : str) {
      if (isalpha(static_cast<unsigned char>(c)) != 0) {
        expected_output_++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_ = 0;
};

namespace {

TEST_P(SakharovARunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    TestType{InType{0, ""}, "empty"},
    TestType{InType{4, "abcd"}, "all_letters"},
    TestType{InType{6, "ab12CD"}, "letters_and_digits"},
    TestType{InType{10, "____123456"}, "no_letters"},
    TestType{InType{5, "AaA1!"}, "mixed_with_punct"},
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SakharovANumberOfLettersMPI, InType>(kTestParam, PPC_SETTINGS_sakharov_a_num_of_letters),
    ppc::util::AddFuncTask<SakharovANumberOfLettersSEQ, InType>(kTestParam, PPC_SETTINGS_sakharov_a_num_of_letters));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SakharovARunFuncTestsProcesses::PrintFuncTestName<SakharovARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SakharovARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sakharov_a_num_of_letters
