#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_max_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_max_elem_matrix {

class PylaevaSMaxElemMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = ppc::util::GetAbsoluteTaskPath(PPC_ID_pylaeva_s_max_elem_matrix, param + ".txt");

    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t rows = 0;
    size_t columns = 0;
    size_t size = 0;
    int max = 0;
    std::vector<int> input;

    file >> rows;
    file >> columns;
    file >> max;

    size = rows * columns;

    input.resize(size);

    for (size_t i = 0; i < size; ++i) {
      if (!(file >> input[i])) {
        throw std::runtime_error("Not enough elements in file");
      }
    }

    input_data_ = std::make_tuple(rows, columns, input);
    expected_data_ = max;

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_data_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_{};
};

namespace {

TEST_P(PylaevaSMaxElemMatrixFuncTests, MaxElemMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"matrix_3x3",       "matrix_5x5",      "matrix_11x11",   "matrix_50x50",
                                             "matrix_100x100",   "matrix_150x100",  "matrix_200x200", "matrix_500x1000",
                                             "matrix_1000x1000", "matrix_1500x1000"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PylaevaSMaxElemMatrixMPI, InType>(kTestParam, PPC_SETTINGS_pylaeva_s_max_elem_matrix),
    ppc::util::AddFuncTask<PylaevaSMaxElemMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_pylaeva_s_max_elem_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PylaevaSMaxElemMatrixFuncTests::PrintFuncTestName<PylaevaSMaxElemMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MaxElemMatrixTests, PylaevaSMaxElemMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace pylaeva_s_max_elem_matrix
