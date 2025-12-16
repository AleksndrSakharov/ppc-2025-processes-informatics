#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "kutergin_a_closest_pair/mpi/include/ops_mpi.hpp"
#include "kutergin_a_closest_pair/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_a_closest_pair {

class KuterginAClosestPairFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KuterginAClosestPairFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(std::get<0>(params));
    expected_ = std::get<1>(std::get<0>(params));
    std::cout << "Test: " << std::get<1>(params) << "\nInput: ";
    for (const auto &v : input_data_) {
      std::cout << v << ' ';
    }
    std::cout << "\nExpected index: " << expected_ << "\n";
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return output_data == expected_;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_ = 0;
};

namespace {
std::vector<int> v1{1, 3, 5, 6, 9};         // closest pair 5-6 => index 2
std::vector<int> v2{10, 11, 30, 40};        // 10-11 => index 0
std::vector<int> v3{5, 2, 3, 10};           // 2-3 => index 1
std::vector<int> v4{100, 99, 101, 200};     // 100-99 => index 0
std::vector<int> v5{1};                     // no pair => -1
std::vector<int> v6{};                      // no pair => -1
std::vector<int> v7{4, 4, 4, 4};            // 4-4 => index 0
std::vector<int> v8{-10, -5, -7, 0};        // closest pair -5,-7 => index 1
std::vector<int> v9{-1, -2, -8, -10, -11};  // closest pair -1,-2 => index 0
std::vector<int> v10{-100, 50, -50, 0};     // closest pair 50,-50 => index 1

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(std::make_tuple(v1, 2), "test1"),  std::make_tuple(std::make_tuple(v2, 0), "test2"),
    std::make_tuple(std::make_tuple(v3, 1), "test3"),  std::make_tuple(std::make_tuple(v4, 0), "test4"),
    std::make_tuple(std::make_tuple(v5, -1), "test5"), std::make_tuple(std::make_tuple(v6, -1), "test6"),
    std::make_tuple(std::make_tuple(v7, 0), "test7"),  std::make_tuple(std::make_tuple(v8, 1), "test8"),
    std::make_tuple(std::make_tuple(v9, 0), "test9"),  std::make_tuple(std::make_tuple(v10, 2), "test10")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KuterginAClosestPairMPI, InType>(kTestParam, PPC_SETTINGS_kutergin_a_closest_pair),
    ppc::util::AddFuncTask<KuterginAClosestPairSEQ, InType>(kTestParam, PPC_SETTINGS_kutergin_a_closest_pair));

const auto kGTestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kNameGen = KuterginAClosestPairFuncTests::PrintFuncTestName<KuterginAClosestPairFuncTests>;

INSTANTIATE_TEST_SUITE_P(ClosestPairTests, KuterginAClosestPairFuncTests, kGTestValues, kNameGen);

TEST_P(KuterginAClosestPairFuncTests, ClosestPairTest) {
  ExecuteTest(GetParam());
}
}  // namespace

}  // namespace kutergin_a_closest_pair
