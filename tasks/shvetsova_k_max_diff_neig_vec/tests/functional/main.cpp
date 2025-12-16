#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "shvetsova_k_max_diff_neig_vec/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_max_diff_neig_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

class ShvetsovaKMaxDiffNeigVecRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_shvetsova_k_max_diff_neig_vec, param + ".txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      std::cerr << "ERROR: Cannot open file: " << abs_path << '\n';
      return;
    }
    file >> expect_res_.first;
    file >> expect_res_.second;
    double num = 0.0;
    std::vector<double> vec;
    while (file >> num) {
      vec.push_back(num);
    }
    input_data_ = vec;
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 0.05;

    return (std::abs(output_data.first - expect_res_.first) <= eps && output_data.second == expect_res_.second);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expect_res_;
};

namespace {

TEST_P(ShvetsovaKMaxDiffNeigVecRunFuncTestsProcesses, DataFromTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShvetsovaKMaxDiffNeigVecMPI, InType>(kTestParam, PPC_SETTINGS_shvetsova_k_max_diff_neig_vec),
    ppc::util::AddFuncTask<ShvetsovaKMaxDiffNeigVecSEQ, InType>(kTestParam,
                                                                PPC_SETTINGS_shvetsova_k_max_diff_neig_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShvetsovaKMaxDiffNeigVecRunFuncTestsProcesses::PrintFuncTestName<ShvetsovaKMaxDiffNeigVecRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MaxDiffTest, ShvetsovaKMaxDiffNeigVecRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shvetsova_k_max_diff_neig_vec
