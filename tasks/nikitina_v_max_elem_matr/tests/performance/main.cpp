#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <random>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"
#include "nikitina_v_max_elem_matr/mpi/include/ops_mpi.hpp"
#include "nikitina_v_max_elem_matr/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitina_v_max_elem_matr {

class NikitinaVMaxElemMatrPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int rows = 12000;
    int cols = 12000;
    int max_val = std::numeric_limits<int>::min();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-1000, 1000);

    InType generated_matr(2 + (static_cast<size_t>(rows) * cols));
    generated_matr[0] = rows;
    generated_matr[1] = cols;

    for (int i = 0; i < rows * cols; ++i) {
      generated_matr[i + 2] = distrib(gen);
      if (i == 0 || generated_matr[i + 2] > max_val) {
        max_val = generated_matr[i + 2];
      }
    }

    input_data_ = generated_matr;
    expected_output_ = max_val;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return (output_data == expected_output_);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

TEST_P(NikitinaVMaxElemMatrPerfTests, RunPerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MaxElementMatrSEQ, MaxElementMatrMPI>(PPC_SETTINGS_nikitina_v_max_elem_matr);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(NikitinaV_MaxElementMatr_Perf, NikitinaVMaxElemMatrPerfTests, kGtestValues,
                         NikitinaVMaxElemMatrPerfTests::CustomPerfTestName);

}  // namespace nikitina_v_max_elem_matr
