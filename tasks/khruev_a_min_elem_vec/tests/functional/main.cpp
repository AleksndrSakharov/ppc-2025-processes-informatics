#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <climits>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "khruev_a_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace khruev_a_min_elem_vec {

class KhruevAMinElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    // size_t size = 10;
    // for (size_t i = 1; i <= size; i++) {
    // input_data_.push_back(i);
    // }
    // expected_ = 1;
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::cout << "SetUp for " << std::get<1>(params) << '\n';
    input_data_ = std::vector(std::get<0>(std::get<0>(params)));
    std::cout << "We set input data:  \n";
    for (auto x : input_data_) {
      std::cout << x << ' ';
    }
    std::cout << '\n';
    // expected_ = std::get<1>(std::get<0>(params));
    std::cout << "We set expected result:  " << std::get<1>(std::get<0>(params)) << '\n';
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::cout << "CheckTestOutputData for "
              << std::get<1>(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()))
              << '\n';
    return (std::get<1>(std::get<0>(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(
                GetParam()))) == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  // OutType expected_;
};

namespace {

TEST_P(KhruevAMinElemVecFuncTests, MinElemVecc) {
  ExecuteTest(GetParam());
}
std::vector<int> instanse1{1, 2, 3, 4, 5, 6};
std::vector<int> instanse2{2, 2, -13, 4, 1, 5, 6};
std::vector<int> instanse3{-100, 2, 30, 4, 12, 6, 1, 6, 2};
std::vector<int> instanse4{1, 2, 3, 4, 5, 6, 9, 1, 1, 0};
std::vector<int> instanse5{1, 2, 3, 4};
std::vector<int> instanse6{2, 5, 2, 0, -10};
std::vector<int> instanse7{1, 2};
std::vector<int> instanse8{-1, -2, -8, -10, -11};
std::vector<int> instanse9{1, 1, 1, 1, 1};
std::vector<int> instanse10{};

const std::array<TestType, 10> kTestParam = {std::make_tuple(std::make_tuple(instanse1, 1), "test1"),
                                             std::make_tuple(std::make_tuple(instanse2, -13), "test2"),
                                             std::make_tuple(std::make_tuple(instanse3, -100), "test3"),
                                             std::make_tuple(std::make_tuple(instanse4, 0), "test4"),
                                             std::make_tuple(std::make_tuple(instanse5, 1), "test5"),
                                             std::make_tuple(std::make_tuple(instanse6, -10), "test6"),
                                             std::make_tuple(std::make_tuple(instanse7, 1), "test7"),
                                             std::make_tuple(std::make_tuple(instanse8, -11), "test8"),
                                             std::make_tuple(std::make_tuple(instanse9, 1), "test9"),
                                             std::make_tuple(std::make_tuple(instanse10, INT_MAX), "test10")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KhruevAMinElemVecMPI, InType>(kTestParam, PPC_SETTINGS_khruev_a_min_elem_vec),
    ppc::util::AddFuncTask<KhruevAMinElemVecSEQ, InType>(kTestParam, PPC_SETTINGS_khruev_a_min_elem_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KhruevAMinElemVecFuncTests::PrintFuncTestName<KhruevAMinElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinElemVec, KhruevAMinElemVecFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace khruev_a_min_elem_vec
