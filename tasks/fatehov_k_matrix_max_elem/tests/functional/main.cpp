#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"
#include "fatehov_k_matrix_max_elem/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace fatehov_k_matrix_max_elem {

class FatehovKRunFuncTestsMatrixMaxElem : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string out = std::to_string(std::get<0>(test_param)) + "_matrix_" + std::to_string(std::get<1>(test_param)) +
                      "x" + std::to_string(std::get<2>(test_param));
    std::ranges::replace(out, '-', 'm');
    std::ranges::replace(out, '.', '_');
    return out;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t rows = std::get<1>(params);
    size_t columns = std::get<2>(params);
    std::vector<double> matrix = std::get<3>(params);
    input_data_ = std::make_tuple(rows, columns, matrix);
    expected_result_ = std::get<4>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = std::make_tuple(0, 0, std::vector<double>{});
  OutType expected_result_ = 0;
};

namespace {

TEST_P(FatehovKRunFuncTestsMatrixMaxElem, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(1, 3, 4, std::vector<double>{1.3, 2.4, 3.1, 4, 5.0, 6.2, 7, 8, 9, 10, 11, 12}, 12),
    std::make_tuple(2, 3, 3, std::vector<double>{-10.3, -9.1, -8.5, -7.1, -6, -5, -4.0, -3, -2}, -2),
    std::make_tuple(3, 5, 5, std::vector<double>{10000,      124124, 65789, 75445, 4123412, 1412412, 56,  65,  -2,
                                                 -151234124, 63124,  454,   223,   454,     232,     565, 232, 7878,
                                                 2324,       57546,  12412, 454,   434,     466,     444},
                    4123412)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FatehovKMatrixMaxElemMPI, InType>(kTestParam, PPC_SETTINGS_fatehov_k_matrix_max_elem),
    ppc::util::AddFuncTask<FatehovKMatrixMaxElemSEQ, InType>(kTestParam, PPC_SETTINGS_fatehov_k_matrix_max_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FatehovKRunFuncTestsMatrixMaxElem::PrintFuncTestName<FatehovKRunFuncTestsMatrixMaxElem>;

INSTANTIATE_TEST_SUITE_P(TestMatrixMax, FatehovKRunFuncTestsMatrixMaxElem, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fatehov_k_matrix_max_elem
