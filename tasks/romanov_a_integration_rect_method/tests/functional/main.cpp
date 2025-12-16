#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <string>
#include <tuple>

#include "romanov_a_integration_rect_method/common/include/common.hpp"
#include "romanov_a_integration_rect_method/mpi/include/ops_mpi.hpp"
#include "romanov_a_integration_rect_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanov_a_integration_rect_method {

class RomanovAIntegrationRectMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto [f, a, b, n, result] = test_param;

    // Минус (-) от отрицательных чисел в имени gtest нельзя, функцию в имя gtest не поместить, точку (.) тоже... Как
    // жить то?
    return "_id" + std::to_string(static_cast<int>(std::abs((result - a + (2.0 * b) + 4.0) * 334))) + "_n" +
           std::to_string(n);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    auto [f, a, b, n, result] = params;

    input_data_ = std::make_tuple(std::function<double(double)>(f), a, b, n);

    expected_ = result;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return IsEqual(static_cast<double>(output_data), expected_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_ = 0.0;
};

namespace {

TEST_P(RomanovAIntegrationRectMethodFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple([](double x) { return x; }, -1.0, 1.0, 1, 0),
    std::make_tuple([](double x) { return x; }, 0.0, 1.0, 2, 0.5),
    std::make_tuple([](double x) { return x * x; }, 0.0, 1.0, 1000000 + 1, 1.0 / 3.0),
    std::make_tuple([](double x) { return (std::sin(x) * x) - std::sqrt(x); }, 0.0, std::numbers::pi, 1000000 + 2,
                    -0.570626),
    std::make_tuple([](double x) { return std::exp(x) - x; }, -1.0, 3.0, 1000000 + 3,
                    -4.0 - (1.0 / std::numbers::e) + std::exp(3.0)),
    std::make_tuple([](double x) { return std::cos(x) * std::sqrt(x); }, 0.0, std::numbers::pi, 1000000 + 4,
                    -0.894831)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RomanovAIntegrationRectMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_romanov_a_integration_rect_method),
                                           ppc::util::AddFuncTask<RomanovAIntegrationRectMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_romanov_a_integration_rect_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    RomanovAIntegrationRectMethodFuncTests::PrintFuncTestName<RomanovAIntegrationRectMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RomanovAIntegrationRectMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace romanov_a_integration_rect_method
