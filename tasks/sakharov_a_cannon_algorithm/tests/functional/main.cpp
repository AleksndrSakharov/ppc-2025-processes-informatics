#include <gtest/gtest.h>

#include <cmath>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"
#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sakharov_a_cannon_algorithm {

class SakharovARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    InType in = std::get<0>(params);
    input_data_ = in;
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_output_.size() != output_data.size()) {
      return false;
    }

    constexpr double kEps = 1e-9;
    for (std::size_t i = 0; i < expected_output_.size(); ++i) {
      if (std::abs(expected_output_[i] - output_data[i]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

OutType NaiveMultiply(const InType &input) {
  const int n = input.size;
  OutType result(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      const double a_val = input.a[Offset(n, i, k)];
      for (int j = 0; j < n; ++j) {
        result[Offset(n, i, j)] += a_val * input.b[Offset(n, k, j)];
      }
    }
  }

  return result;
}

TestType MakeCase(const InType &input, const std::string &name) {
  return TestType{input, NaiveMultiply(input), name};
}

TEST_P(SakharovARunFuncTestsProcesses, MatrixMultiply) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    MakeCase(InType{1, {2.0}, {3.0}}, "single_element"),
    MakeCase(InType{2, {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}}, "two_by_two"),
    MakeCase(InType{3,
                    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                    {4.0, 1.0, 2.0, 0.0, 3.0, 5.0, -1.0, 7.0, 2.0}},
             "identity_multiplies"),
    MakeCase(InType{4,
                    {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
                    {1.0, 0.0, 2.0, -1.0, 3.0, 1.0, 0.0, 2.0, 1.0, 4.0, -2.0, 0.0, 0.0, 1.0, 3.0, 2.0}},
             "four_by_four_blocked")
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SakharovACannonAlgorithmMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sakharov_a_cannon_algorithm),
                                           ppc::util::AddFuncTask<SakharovACannonAlgorithmSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sakharov_a_cannon_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SakharovARunFuncTestsProcesses::PrintFuncTestName<SakharovARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, SakharovARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sakharov_a_cannon_algorithm
