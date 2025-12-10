#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

class MonteCarloFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const auto &full = info.param;
    const TestType &t = std::get<2>(full);
    std::string size_name = std::get<1>(t);
    std::string task_name = std::get<1>(full);
    return task_name + "_" + size_name;
  }

 protected:
  void SetUp() override {
    auto full_param = GetParam();
    TestType t = std::get<2>(full_param);

    int samples = std::get<0>(t);

    input_.a = {0.0};
    input_.b = {1.0};
    input_.samples = samples;
    input_.func = [](const std::vector<double> &x) { return x[0] * x[0]; };
  }

  bool CheckTestOutputData(OutType &out) final {
    return std::abs(out - (1.0 / 3.0)) < 0.05;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
};

TEST_P(MonteCarloFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kParams = {
    std::make_tuple(1000, "small"),
    std::make_tuple(5000, "medium"),
    std::make_tuple(20000, "large"),
};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationSEQ, InType>(kParams, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationMPI, InType>(kParams, PPC_SETTINGS_example_processes));

INSTANTIATE_TEST_SUITE_P(IntegrationTests, MonteCarloFuncTests, ppc::util::ExpandToValues(kTaskList),
                         MonteCarloFuncTests::PrintTestParam);

class MonteCarloExtraFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintName(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const auto &full = info.param;
    const TestType &t = std::get<2>(full);
    std::string task_name = std::get<1>(full);
    return task_name + "_extra_" + std::get<1>(t);
  }

 protected:
  void SetUp() override {
    auto full = GetParam();
    TestType t = std::get<2>(full);
    int scenario = std::get<0>(t);

    switch (scenario) {
      case 0:  // tiny interval
        input_.a = {0.0};
        input_.b = {1e-6};
        input_.samples = 5000;
        break;

      case 1:  // shifted interval
        input_.a = {2.0};
        input_.b = {3.0};
        input_.samples = 8000;
        break;

      case 2:  // samples = 1
        input_.a = {0.0};
        input_.b = {1.0};
        input_.samples = 1;
        break;

      case 3:  // large samples
        input_.a = {0.0};
        input_.b = {1.0};
        input_.samples = 100000;
        break;

      case 4:  // negative interval
        input_.a = {-1.0};
        input_.b = {1.0};
        input_.samples = 20000;
        break;
      default:
        // clang-tidy requires default, but we do nothing
        break;
    }

    input_.func = [](const std::vector<double> &x) { return x[0] * x[0]; };
  }

  bool CheckTestOutputData(OutType &out) final {
    auto full = GetParam();
    int scenario = std::get<0>(std::get<2>(full));

    switch (scenario) {
      case 0:  // tiny interval [0, 1e-6]
        return std::abs(out - (1e-18 / 3.0)) < 1e-10;

      case 1: {  // shifted interval [2, 3]
        double exact = (std::pow(3.0, 3) - std::pow(2.0, 3)) / 3.0;
        return std::abs(out - exact) < 0.2;
      }

      case 2:  // samples=1, любое значение допустимо, главное в пределах диапазона
        return out >= 0.0 && out <= 1.0;

      case 3:  // high accuracy expected
        return std::abs(out - (1.0 / 3.0)) < 0.01;

      case 4:  // integral x^2 on [-1,1] = 2/3
        return std::abs(out - (2.0 / 3.0)) < 0.05;
      default:
        return false;  // unreachable but required by clang-tidy
    }
    return false;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
};

TEST_P(MonteCarloExtraFuncTests, CornerCaseTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kExtraParams = {
    std::make_tuple(0, "tiny_interval"), std::make_tuple(1, "shifted_interval"),  std::make_tuple(2, "samples_1"),
    std::make_tuple(3, "large_samples"), std::make_tuple(4, "negative_interval"),
};

const auto kExtraTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationSEQ, InType>(kExtraParams, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<DorofeevIMonteCarloIntegrationMPI, InType>(kExtraParams, PPC_SETTINGS_example_processes));

INSTANTIATE_TEST_SUITE_P(ExtraIntegrationTests, MonteCarloExtraFuncTests, ppc::util::ExpandToValues(kExtraTaskList),
                         MonteCarloExtraFuncTests::PrintName);

}  // namespace dorofeev_i_monte_carlo_integration_processes
