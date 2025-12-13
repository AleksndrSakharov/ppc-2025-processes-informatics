#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"
#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevIRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType & /*param*/) {
    static int test_counter = 0;
    return "test_" + std::to_string(test_counter++);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KiselevIRunFuncTestsProcesses, MaxInRowsFromMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    std::make_tuple(std::vector<std::vector<int>>{}, std::vector<int>{}),

    std::make_tuple(std::vector<std::vector<int>>{{}, {}, {}},
                    std::vector<int>{std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                                     std::numeric_limits<int>::min()}),

    std::make_tuple(std::vector<std::vector<int>>{}, std::vector<int>{}),

    std::make_tuple(std::vector<std::vector<int>>{{1}, {}, {2}},
                    std::vector<int>{1, std::numeric_limits<int>::min(), 2}),

    std::make_tuple(std::vector<std::vector<int>>{{5}}, std::vector<int>{5}),

    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, std::vector<int>{3, 6, 9}),

    std::make_tuple(std::vector<std::vector<int>>{{-10, -50, -30}, {10, 20, 35}, {0, 0, 0}},
                    std::vector<int>{-10, 35, 0}),

    std::make_tuple(std::vector<std::vector<int>>{{228}}, std::vector<int>{228}),

    std::make_tuple(std::vector<std::vector<int>>{{1}, {2, 3, 4, 5, 6}, {7, 8}}, std::vector<int>{1, 6, 8}),

    std::make_tuple(std::vector<std::vector<int>>{{0, 0, 0}, {5, 5, 5}, {1, 1, 1}}, std::vector<int>{0, 5, 1}),

    std::make_tuple(std::vector<std::vector<int>>{{1000000000, -5, 3}, {7, 800000000, 9}, {-1, -2, -3}},
                    std::vector<int>{1000000000, 800000000, -1}),

    std::make_tuple(std::vector<std::vector<int>>{{9}, {4, 3}, {10, 20, 30, 40}, {-5, -10, -1}},
                    std::vector<int>{9, 4, 40, -1}),

    std::make_tuple(
        std::vector<std::vector<int>>{
            {1, 9, 2},
            {3, 7, 6},
            {0, 100, 50},
        },
        std::vector<int>{9, 7, 100}),

    std::make_tuple(std::vector<std::vector<int>>{{-1, -100, -50}, {0, -1, -2}, {-10, 0, -5}},
                    std::vector<int>{-1, 0, 0}),

    std::make_tuple(
        std::vector<std::vector<int>>{
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 11, 12},
        },
        std::vector<int>{100}),

    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                                  {10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
                                                  {5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                                                  {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10},
                                                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                  {100, 50, 20, 10, 5, 1, 0, -1, -2, -3},
                                                  {7, 14, 21, 28, 35, 42, 49, 56, 63, 70},
                                                  {9, 18, 27, 36, 45, 54, 63, 72, 81, 90},
                                                  {1000, 200, 300, 400, 500, 600, 700, 800, 900, 999},
                                                  {-1000, -2000, -3000, -4000, -5000, -6000, -7000, -8000, -9000, -1}},
                    std::vector<int>{10, 10, 5, -1, 0, 100, 70, 90, 1000, -1})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KiselevITestTaskMPI, InType>(kTestParam, PPC_SETTINGS_kiselev_i_max_value_in_strings),
    ppc::util::AddFuncTask<KiselevITestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_kiselev_i_max_value_in_strings));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KiselevIRunFuncTestsProcesses::PrintFuncTestName<KiselevIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixRowMaxTests, KiselevIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kiselev_i_max_value_in_strings
