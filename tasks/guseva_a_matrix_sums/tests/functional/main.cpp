#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "guseva_a_matrix_sums/common/include/common.hpp"
#include "guseva_a_matrix_sums/mpi/include/ops_mpi.hpp"
#include "guseva_a_matrix_sums/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace guseva_a_matrix_sums {

class GusevaARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string input_data_source =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_guseva_a_matrix_sums, "cases/" + param + ".txt");
    std::string expected_data_source =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_guseva_a_matrix_sums, "expected/" + param + ".txt");

    std::ifstream file(input_data_source);
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<double> inp;
    std::vector<double> exp;
    file >> rows;
    file >> columns;
    int num = 0;
    while (file >> num) {
      inp.push_back(num);
    }
    file.close();
    file = std::ifstream(expected_data_source);
    file >> columns;
    while (file >> num) {
      exp.push_back(num);
    }
    file.close();

    input_data_ = InType(rows, columns, inp);
    expected_data_ = OutType(exp);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    for (uint32_t i = 0; i < expected_data_.size(); i++) {
      if (std::abs(output_data[i] - expected_data_[i]) > kEpsilon) {
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
  OutType expected_data_;
};

namespace {

TEST_P(GusevaARunFuncTestsProcesses, MatrixSumsByColumns) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<guseva_a_matrix_sums::GusevaAMatrixSumsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_guseva_a_matrix_sums),
                                           ppc::util::AddFuncTask<guseva_a_matrix_sums::GusevaAMatrixSumsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_guseva_a_matrix_sums));

inline const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

inline const auto kPerfTestName = GusevaARunFuncTestsProcesses::PrintFuncTestName<GusevaARunFuncTestsProcesses>;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(GusevaAMatrix, GusevaARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_matrix_sums
