#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class MostDiffNeighVecElemsRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto expected_pair = std::get<0>(test_param);
    int first = std::get<0>(expected_pair);
    int second = std::get<1>(expected_pair);

    std::string first_str = (first < 0) ? "neg" + std::to_string(-first) : std::to_string(first);
    std::string second_str = (second < 0) ? "neg" + std::to_string(-second) : std::to_string(second);

    return "expected_" + first_str + "_" + second_str;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<1>(params);
    expected_output_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(MostDiffNeighVecElemsRunFuncTestsProcesses, RunFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(std::make_tuple(1, 10), std::vector<int>{1, 10, 3, 10, 5}),
    std::make_tuple(std::make_tuple(5, 20), std::vector<int>{5, 20, 15, 10}),
    std::make_tuple(std::make_tuple(7, 7), std::vector<int>{7}),
    std::make_tuple(std::make_tuple(0, 100), std::vector<int>{0, 100, 100, 25}),
    std::make_tuple(std::make_tuple(-10, -5), std::vector<int>{-10, -5, -1, 3, 0, 0}),
    std::make_tuple(std::make_tuple(2, 3), std::vector<int>{2, 2, 3}),
    std::make_tuple(std::make_tuple(10, 1), std::vector<int>{10, 1, 5, 3})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LeonovaAMostDiffNeighVecElemsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems),
                                           ppc::util::AddFuncTask<LeonovaAMostDiffNeighVecElemsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MostDiffNeighVecElemsRunFuncTestsProcesses::PrintFuncTestName<MostDiffNeighVecElemsRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RunVecFunc, MostDiffNeighVecElemsRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace leonova_a_most_diff_neigh_vec_elems
