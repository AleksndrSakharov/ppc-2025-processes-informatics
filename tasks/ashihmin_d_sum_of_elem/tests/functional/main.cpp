// main(functional.cpp)
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>
#include <utility>

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"
#include "ashihmin_d_sum_of_elem/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_sum_of_elem/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_sum_of_elem {

class AshihminDElemVecSumFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    int len = std::get<0>(std::get<2>(GetParam()));
    input_data_.assign(len, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::cmp_equal(output_data, static_cast<OutType>(input_data_.size()));
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(AshihminDElemVecSumFuncTest, RunTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kParams = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<AshihminDElemVecsSumMPI, InType>(kParams, PPC_SETTINGS_ashihmin_d_sum_of_elem),
    ppc::util::AddFuncTask<AshihminDElemVecsSumSEQ, InType>(kParams, PPC_SETTINGS_ashihmin_d_sum_of_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AshihminDElemVecSumFuncTest::PrintFuncTestName<AshihminDElemVecSumFuncTest>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, AshihminDElemVecSumFuncTest, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace ashihmin_d_sum_of_elem
