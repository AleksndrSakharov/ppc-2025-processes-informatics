#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include "pikhotskiy_r_elem_vec_sum/common/include/common.hpp"
#include "pikhotskiy_r_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pikhotskiy_r_elem_vec_sum {
class VectorSumTestSuite : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType test_case = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_values_ = std::get<0>(test_case);
    expected_output_ = std::get<2>(test_case);
  }

  // NOLINTNEXTLINE(readability-make-member-function-const)
  bool CheckTestOutputData(OutType &computed_result) final {
    return computed_result == expected_output_;
  }

  InType GetTestInputData() final {
    return input_values_;
  }

 private:
  InType input_values_{0, {}};
  OutType expected_output_{0};
};

namespace {

TEST_P(VectorSumTestSuite, CheckVectorSumComputation) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestScenarios = {
    // Базовые тесты
    std::make_tuple(std::make_tuple(0, std::vector<int>{}), "empty_vector", 0LL),
    std::make_tuple(std::make_tuple(1, std::vector<int>{5}), "single_element", 5LL),
    std::make_tuple(std::make_tuple(3, std::vector<int>{1, 2, 3}), "small_positive", 6LL),

    // Тесты с нулями
    std::make_tuple(std::make_tuple(3, std::vector<int>{0, 0, 0}), "all_zeros", 0LL),
    std::make_tuple(std::make_tuple(4, std::vector<int>{1, 0, -1, 0}), "mixed_with_zeros", 0LL),

    // Тесты с отрицательными числами
    std::make_tuple(std::make_tuple(3, std::vector<int>{-1, -2, -3}), "all_negative", -6LL),
    std::make_tuple(std::make_tuple(3, std::vector<int>{1, -2, 3}), "mixed_signs", 2LL),
    std::make_tuple(std::make_tuple(4, std::vector<int>{1, -1, 2, -2}), "balanced_positive_negative", 0LL),

    // Граничные значения
    std::make_tuple(std::make_tuple(2, std::vector<int>{2147483646, 1}), "near_int_max", 2147483647LL),
    std::make_tuple(std::make_tuple(2, std::vector<int>{-2147483647, -1}), "near_int_min", -2147483648LL),
    std::make_tuple(std::make_tuple(2, std::vector<int>{2147483646, 2147483647}), "large_positive_sum", 4294967293LL),

    // Большие вектора (проверка производительности в функциональных тестах)
    std::make_tuple(std::make_tuple(5, std::vector<int>{10, 20, 30, 40, 50}), "medium_vector", 150LL),
    std::make_tuple(std::make_tuple(6, std::vector<int>{1, 3, 5, 7, 9, 11}), "odd_numbers", 36LL),

    // Специфические случаи
    std::make_tuple(std::make_tuple(4, std::vector<int>{1, 2, 4, 8}), "powers_of_two", 15LL)};

const auto kTestCombinations =
    std::tuple_cat(ppc::util::AddFuncTask<pikhotskiy_r_elem_vec_sum::PikhotskiyRElemVecSumMPI, InType>(
                       kTestScenarios, PPC_SETTINGS_pikhotskiy_r_elem_vec_sum),
                   ppc::util::AddFuncTask<pikhotskiy_r_elem_vec_sum::PikhotskiyRElemVecSumSEQ, InType>(
                       kTestScenarios, PPC_SETTINGS_pikhotskiy_r_elem_vec_sum));

const auto kGtestInstances = ppc::util::ExpandToValues(kTestCombinations);

const auto kTestNameGenerator = VectorSumTestSuite::PrintFuncTestName<VectorSumTestSuite>;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(VectorSumTests, VectorSumTestSuite, kGtestInstances, kTestNameGenerator);

}  // namespace

}  // namespace pikhotskiy_r_elem_vec_sum
