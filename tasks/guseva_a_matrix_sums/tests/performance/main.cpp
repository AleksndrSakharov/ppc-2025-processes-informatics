#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "guseva_a_matrix_sums/common/include/common.hpp"
#include "guseva_a_matrix_sums/mpi/include/ops_mpi.hpp"
#include "guseva_a_matrix_sums/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace guseva_a_matrix_sums {

class GusevaAMatrixSumsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{0, 0, {}};
  OutType expected_data_;

  void SetUp() override {
    std::string input_data_source = ppc::util::GetAbsoluteTaskPath(PPC_ID_guseva_a_matrix_sums, "perf/input.txt");
    std::string expected_data_source = ppc::util::GetAbsoluteTaskPath(PPC_ID_guseva_a_matrix_sums, "perf/expected.txt");

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

  [[nodiscard]] InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(GusevaAMatrixSumsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GusevaAMatrixSumsMPI, GusevaAMatrixSumsSEQ>(PPC_SETTINGS_guseva_a_matrix_sums);

inline const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

inline const auto kPerfTestName = GusevaAMatrixSumsRunPerfTestProcesses::CustomPerfTestName;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(GusevaPerf, GusevaAMatrixSumsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_matrix_sums
