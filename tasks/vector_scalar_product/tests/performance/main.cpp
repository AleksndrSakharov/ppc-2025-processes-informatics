#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "vector_scalar_product/common/include/common.hpp"
#include "vector_scalar_product/mpi/include/ops_mpi.hpp"
#include "vector_scalar_product/seq/include/ops_seq.hpp"

namespace vector_scalar_product {

class VectorScalarProductPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr std::size_t kSize = 20000000;
    input_.lhs.resize(kSize);
    input_.rhs.resize(kSize);
    for (std::size_t i = 0; i < kSize; ++i) {
      input_.lhs[i] = static_cast<double>(i % 97) * 0.5;
      input_.rhs[i] = static_cast<double>((i * 3) % 101) * 0.25;
    }
    expected_ = std::inner_product(input_.lhs.begin(), input_.lhs.end(), input_.rhs.begin(), 0.0);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_) < 1e-6;
  }

 private:
  InType input_{};
  double expected_ = 0.0;
};

namespace {

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, VectorScalarProductMpi, VectorScalarProductSeq>(
    PPC_SETTINGS_vector_scalar_product);

const auto kGTestValues = ppc::util::TupleToGTestValues(kPerfTasks);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(DotProductPerf, VectorScalarProductPerfTests, kGTestValues,
                         VectorScalarProductPerfTests::CustomPerfTestName);

TEST_P(VectorScalarProductPerfTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace vector_scalar_product
