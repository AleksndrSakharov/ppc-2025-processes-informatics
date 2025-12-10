#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"
#include "lukin_i_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "lukin_i_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lukin_i_elem_vec_sum {

class LukinIRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string test_name = std::get<1>(test_param);
    return test_name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    auto vec_size = static_cast<int>(std::get<0>(params));
    input_data_ = std::vector<int>(vec_size, 1);

    expected_ = static_cast<OutType>(vec_size);
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

TEST_P(LukinIRunFuncTestsProcesses, ElemVecSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(1, "single_element"), std::make_tuple(2, "two_elements"),   std::make_tuple(3, "three_elements"),
    std::make_tuple(4, "four_elements"),  std::make_tuple(5, "five_elements"),  std::make_tuple(6, "six_elements"),
    std::make_tuple(7, "seven_elements"), std::make_tuple(8, "eight_elements"), std::make_tuple(9, "nine_elements"),
    std::make_tuple(10, "medium_size"),   std::make_tuple(11, "prime_size"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LukinIElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum),
                   ppc::util::AddFuncTask<LukinIElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProcesses::PrintFuncTestName<LukinIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ElemVecTests, LukinIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_elem_vec_sum
