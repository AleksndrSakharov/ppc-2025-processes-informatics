#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"
#include "votincev_d_matrix_mult/mpi/include/ops_mpi.hpp"
#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  InType input_data;
  OutType expected_res;
  void SetUp() override {
    TestType param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string input_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_votincev_d_matrix_mult, param + ".txt");

    std::ifstream file(input_path);
    if (!file.is_open()) {
      return;
    }

    size_t param_m = 0;
    size_t param_n = 0;
    size_t param_k = 0;
    file >> param_m >> param_n >> param_k;

    std::vector<double> matrix_a(param_m * param_k);
    std::vector<double> matrix_b(param_k * param_n);

    for (double &v : matrix_a) {
      file >> v;
    }
    for (double &v : matrix_b) {
      file >> v;
    }

    input_data = std::make_tuple(param_m, param_n, param_k, matrix_a, matrix_b);

    // вычисляю предполагаемый результат
    expected_res.assign(param_m * param_n, 0.0);
    for (size_t i = 0; i < param_m; ++i) {
      for (size_t j = 0; j < param_n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < param_k; ++k) {
          sum += matrix_a[(i * param_k) + k] * matrix_b[(k * param_n) + j];
        }
        expected_res[(i * param_n) + j] = sum;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 1,2... процессы не владеют нужным результатом
    if (output_data.size() != expected_res.size()) {
      return true;
    }
    // 0й процесс должен иметь корректную матрицу после умножения
    return output_data == expected_res;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

namespace {

TEST_P(VotincevDMatrixMultRunFuncTestsProcesses, MatrixMultiplicationTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VotincevDMatrixMultMPI, InType>(kTestParam, PPC_SETTINGS_votincev_d_matrix_mult),
    ppc::util::AddFuncTask<VotincevDMatrixMultSEQ, InType>(kTestParam, PPC_SETTINGS_votincev_d_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    VotincevDMatrixMultRunFuncTestsProcesses::PrintFuncTestName<VotincevDMatrixMultRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, VotincevDMatrixMultRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace votincev_d_matrix_mult
