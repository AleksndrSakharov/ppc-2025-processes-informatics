#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "smyshlaev_a_mat_mul/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_mat_mul/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_mat_mul {

class SmyshlaevAMatMulRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<4>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::make_tuple(std::get<0>(params), std::get<1>(params), std::get<2>(params), std::get<3>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int m = std::get<0>(input_data_);
    const auto &mat_a = std::get<1>(input_data_);
    int k = std::get<2>(input_data_);
    const auto &mat_b = std::get<3>(input_data_);
    int n = static_cast<int>(mat_b.size()) / k;

    std::vector<double> expected(static_cast<size_t>(m) * n, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int idx = 0; idx < k; ++idx) {
          expected[(i * n) + j] += mat_a[(i * k) + idx] * mat_b[(idx * n) + j];
        }
      }
    }

    if (output_data.size() != expected.size()) {
      return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::abs(output_data[i] - expected[i]) > 1e-5) {
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
};

namespace {

TEST_P(SmyshlaevAMatMulRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}
TestType CreateRandomTest(int m, int n, int k, const std::string &name) {
  std::vector<double> a(static_cast<size_t>(m) * k);
  std::vector<double> b(static_cast<size_t>(k) * n);
  for (int i = 0; i < m * k; ++i) {
    a[i] = (i % 10) + 1.0;
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = (i % 10) + 1.0;
  }

  return std::make_tuple(m, a, k, b, name);
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(1, std::vector<double>{2.0}, 1, std::vector<double>{3.0}, "SingleElement"),

    std::make_tuple(2, std::vector<double>{1.0, 2.0, 3.0, 4.0}, 2, std::vector<double>{1.0, 0.0, 0.0, 1.0},
                    "IdentityMatrixMultiplication"),

    std::make_tuple(2, std::vector<double>{1.0, 2.0, 3.0, 4.0}, 2, std::vector<double>{0.0, 0.0, 0.0, 0.0},
                    "ZeroMatrixMultiplication"),

    CreateRandomTest(2, 3, 2, "Rectangular_2x3_x_3x2"),

    CreateRandomTest(3, 2, 3, "Rectangular_3x2_x_2x3"),

    CreateRandomTest(1, 5, 5, "RowVector_x_Matrix"),

    CreateRandomTest(5, 5, 1, "Matrix_x_ColVector"),

    CreateRandomTest(10, 10, 10, "Square_10x10"),

    CreateRandomTest(7, 5, 3, "PrimeDimensions_7x5_x_5x3"),

    CreateRandomTest(2, 2, 10, "Wide_B_Matrix"),

    CreateRandomTest(10, 1, 10, "ForceSeq_10_x_1_x_1_10"),

    CreateRandomTest(32, 32, 32, "Square_32x32_LoadTest")};
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SmyshlaevAMatMulMPI, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_mat_mul),
                   ppc::util::AddFuncTask<SmyshlaevAMatMulSEQ, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_mat_mul));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SmyshlaevAMatMulRunFuncTestsProcesses::PrintFuncTestName<SmyshlaevAMatMulRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatMulTests, SmyshlaevAMatMulRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace smyshlaev_a_mat_mul
