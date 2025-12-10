#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"
#include "kurpiakov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_elem_vec_sum {
class KurpiakovAElemVecSumFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_data_ = std::get<2>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_data_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{0, {}};
  OutType expected_data_{0};
};

namespace {
TEST_P(KurpiakovAElemVecSumFuncTest, ElemVecSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(std::make_tuple(0, std::vector<int>{}), "test1_empty", 0LL),
    std::make_tuple(std::make_tuple(1, std::vector<int>{5}), "test2_single", 5LL),
    std::make_tuple(std::make_tuple(3, std::vector<int>{1, 2, 3}), "test3_positive", 6LL),
    std::make_tuple(std::make_tuple(3, std::vector<int>{0, 0, 0}), "test4_zeros", 0LL),
    std::make_tuple(std::make_tuple(3, std::vector<int>{1, -2, 3}), "test5_mixed", 2LL),
    std::make_tuple(std::make_tuple(2, std::vector<int>{2147483646, 1}), "test6_border", 2147483647LL),
    std::make_tuple(std::make_tuple(2, std::vector<int>{2147483646, 2147483647}), "test7_overflow", 4294967293LL),
    std::make_tuple(std::make_tuple(3, std::vector<int>{-1, -2, -3}), "test8_negative", -6LL),
    std::make_tuple(std::make_tuple(4, std::vector<int>{1, -1, 2, -2}), "test9_alternating", 0LL)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<kurpiakov_a_elem_vec_sum::KurpiakovAElemVecSumMPI, InType>(
                       kTestParam, PPC_SETTINGS_kurpiakov_a_elem_vec_sum),
                   ppc::util::AddFuncTask<kurpiakov_a_elem_vec_sum::KurpiakovAElemVecSumSEQ, InType>(
                       kTestParam, PPC_SETTINGS_kurpiakov_a_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KurpiakovAElemVecSumFuncTest::PrintFuncTestName<KurpiakovAElemVecSumFuncTest>;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(KurpiakovAVec, KurpiakovAElemVecSumFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kurpiakov_a_elem_vec_sum
