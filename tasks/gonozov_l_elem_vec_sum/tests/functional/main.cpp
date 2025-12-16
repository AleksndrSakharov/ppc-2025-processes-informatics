#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"
#include "gonozov_l_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "gonozov_l_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_elem_vec_sum {

class GonozovLRunElemVecSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::vector<int>(static_cast<int>(std::get<0>(params)), 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return output_data == static_cast<OutType>(std::get<0>(params));
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(GonozovLRunElemVecSumFuncTests, ElemVecSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(1, "Vector_consisting_of_a_single_element"),
                                            std::make_tuple(3, "Vector_consisting_of_three_elements"),
                                            std::make_tuple(5, "Vector_consisting_of_five_elements"),
                                            std::make_tuple(10, "Vector_consisting_of_ten_elements"),
                                            std::make_tuple(15, "Vector_consisting_of_fifteen_elements"),
                                            std::make_tuple(20, "Vector_consisting_of_twenty_elements")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GonozovLElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_gonozov_l_elem_vec_sum),
    ppc::util::AddFuncTask<GonozovLElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_gonozov_l_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GonozovLRunElemVecSumFuncTests::PrintFuncTestName<GonozovLRunElemVecSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(ElemVecTests, GonozovLRunElemVecSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gonozov_l_elem_vec_sum
