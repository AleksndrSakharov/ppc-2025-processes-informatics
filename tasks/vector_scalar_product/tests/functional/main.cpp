#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vector_scalar_product/common/include/common.hpp"
#include "vector_scalar_product/mpi/include/ops_mpi.hpp"
#include "vector_scalar_product/seq/include/ops_seq.hpp"

namespace vector_scalar_product {

struct DotProductCase {
  std::vector<double> lhs;
  std::vector<double> rhs;
  double expected = 0.0;
};

using TestType = DotProductCase;

class VectorScalarProductFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    const auto size_tag = std::to_string(param.lhs.size());
    const auto value_tag = std::to_string(static_cast<int>(param.expected));
    return size_tag + "_" + value_tag;
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_.lhs = param.lhs;
    input_.rhs = param.rhs;
    expected_ = param.expected;
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(output_data - expected_) < 1e-9;
  }

 private:
  InType input_{};
  double expected_ = 0.0;
};

namespace {

constexpr double kPi = std::numbers::pi;

const std::array<TestType, 3> kTestParams = {TestType{{1.0, 2.0, 3.0}, {4.0, -5.0, 6.0}, 12.0},
                                             TestType{{0.5, 1.5, -2.0, 4.0}, {2.0, 1.0, -1.0, 0.0}, 4.5},
                                             TestType{{kPi, 0.0, -kPi}, {1.0, 2.0, -1.0}, 2.0 * kPi}};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<VectorScalarProductSeq, InType>(kTestParams, PPC_SETTINGS_vector_scalar_product),
    ppc::util::AddFuncTask<VectorScalarProductMpi, InType>(kTestParams, PPC_SETTINGS_vector_scalar_product));

const auto kGTestValues = ppc::util::ExpandToValues(kTaskList);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(DotProduct, VectorScalarProductFuncTests, kGTestValues,
                         VectorScalarProductFuncTests::PrintFuncTestName<VectorScalarProductFuncTests>);

TEST_P(VectorScalarProductFuncTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace vector_scalar_product
