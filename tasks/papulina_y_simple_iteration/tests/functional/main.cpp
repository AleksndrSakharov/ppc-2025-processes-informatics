#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "papulina_y_simple_iteration/common/include/common.hpp"
#include "papulina_y_simple_iteration/mpi/include/ops_mpi.hpp"
#include "papulina_y_simple_iteration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace papulina_y_simple_iteration {

class PapulinaYSimpleIterationRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string data_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_papulina_y_simple_iteration, params + ".txt");
    std::ifstream file(data_path);
    if (!file.is_open()) {
      std::cout << "Cannot open file: " << data_path;
    }
    size_t n = 0;
    file >> n;
    std::vector<double> a(n * n);
    for (size_t i = 0; i < n * n; i++) {
      char comma = 0;
      file >> a[i];
      if (i < (n * n) - 1) {
        file >> comma;
      }
    }
    std::vector<double> b(n);
    for (size_t i = 0; i < n; i++) {
      char comma = 0;
      file >> b[i];
      if (i < n - 1) {
        file >> comma;
      }
    }
    std::vector<double> expected_x(n);
    for (size_t i = 0; i < n; i++) {
      char comma = 0;
      file >> expected_x[i];
      if (i < n - 1) {
        file >> comma;
      }
    }
    file.close();

    input_data_ = std::make_tuple(n, a, b);
    expectedResult_ = expected_x;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expectedResult_.size() != output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); i++) {
      if (std::abs(expectedResult_[i] - output_data[i]) > eps_) {
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
  OutType expectedResult_;
  double eps_ = 1e-6;
};

namespace {

TEST_P(PapulinaYSimpleIterationRunFuncTestsProcesses, SimpleIteration) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 16> kTestParam = {"DiagonalDominant3x3",
                                             "PureDiagonal3x3",
                                             "Tridiagonal4x4",
                                             "WithNegativeElements2x2",
                                             "SingleEquation1x1",
                                             "DifferentDiagonals3x3",
                                             "LargeDiagonal4x4",
                                             "NonSymmetric4x4",
                                             "System5x5",
                                             "System8x8",

                                             "AlternatingSigns3x3",
                                             "Matrix10x10",
                                             "AverageConvergence4x4",
                                             "FastConvergence4x4",
                                             "SpecificTest5x5",
                                             "AlternatingCharacters4x4"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PapulinaYSimpleIterationMPI, InType>(kTestParam, PPC_SETTINGS_papulina_y_simple_iteration),
    ppc::util::AddFuncTask<PapulinaYSimpleIterationSEQ, InType>(kTestParam, PPC_SETTINGS_papulina_y_simple_iteration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PapulinaYSimpleIterationRunFuncTestsProcesses::PrintFuncTestName<PapulinaYSimpleIterationRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SimpleIterationTests, PapulinaYSimpleIterationRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

TEST(PapulinaYSimpleIterationValidation_seq_, NonDiagonalDominant) {  // NOLINT
  InType invalid_input = std::make_tuple(3,
                                         std::vector<double>{
                                             1.0, 5.0, 5.0,  // |1| < |5| + |5| = 10
                                             5.0, 1.0, 5.0,  // |1| < |5| + |5| = 10
                                             5.0, 5.0, 1.0   // |1| < |5| + |5| = 10
                                         },
                                         std::vector<double>{1.0, 1.0, 1.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}

TEST(PapulinaYSimpleIterationValidation_seq_, SingularMatrix) {  // NOLINT
  InType invalid_input = std::make_tuple(2, std::vector<double>{1.0, 2.0, 2.0, 4.0}, std::vector<double>{3.0, 6.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, DetermCheckingIsWorking) {  // NOLINT
  InType invalid_input = std::make_tuple(2,
                                         std::vector<double>{
                                             1.0, 2.0, 2.0, 4.0  // det = 1*4 - 2*2 = 0
                                         },
                                         std::vector<double>{3.0, 6.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.GetDetermCheckingResult(std::get<1>(invalid_input), std::get<0>(invalid_input)));
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, DetermCheckingForMatrix5x5IsWorking) {  // NOLINT
  InType invalid_input =
      std::make_tuple(5, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 2.0,  3.0,  4.0, 5.0, 6.0, 3.0, 4.0, 5.0,
                                             6.0, 7.0, 5.0, 7.0, 9.0, 11.0, 13.0, 2.0, 4.0, 6.0, 8.0, 10.0},
                      std::vector<double>{15.0, 20.0, 25.0, 45.0, 30.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.GetDetermCheckingResult(std::get<1>(invalid_input), std::get<0>(invalid_input)));
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, DiagonalDominanceCheckingIsWorking) {  // NOLINT
  InType invalid_input = std::make_tuple(3,
                                         std::vector<double>{1.0, 5.0, 5.0,  // |1| < |5| + |5| = 10
                                                             5.0, 1.0, 5.0, 5.0, 5.0, 1.0},
                                         std::vector<double>{1.0, 1.0, 1.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.GetDiagonalDominanceResult(std::get<1>(invalid_input), std::get<0>(invalid_input)));
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, DiagonalDominanceCheckingForMatrix5x5IsWorking) {  // NOLINT
  InType invalid_input = std::make_tuple(
      5, std::vector<double>{10.0, -1.0, -2.0, -3.0, -4.0, -10.0, 5.0,  -1.0, -2.0, -3.0, -4.0, -5.0, 8.0,
                             -1.0, -2.0, -3.0, -4.0, -5.0, 12.0,  -1.0, -2.0, -3.0, -4.0, -5.0, 20.0},
      std::vector<double>{0.0, -11.0, -4.0, -1.0, 6.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.GetDiagonalDominanceResult(std::get<1>(invalid_input), std::get<0>(invalid_input)));
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, TryToUseMatrixWithZeroRows) {  // NOLINT
  InType invalid_input = std::make_tuple(0, std::vector<double>(), std::vector<double>());
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, ZeroOnDiagonal) {  // NOLINT
  InType invalid_input = std::make_tuple(3, std::vector<double>{5.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 4.0},
                                         std::vector<double>{8.0, 3.0, 6.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_seq_, RowSwapsToRevealZeroDeterminant) {  // NOLINT
  InType invalid_input = std::make_tuple(
      4,
      std::vector<double>{1e-15, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0,  // строка 2 = 2 * строка 1
                          1e-14, 3.0, 4.0, 5.0},
      std::vector<double>{10.0, 26.0, 52.0, 13.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }

  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_mpi_, SingularMatrix) {  // NOLINT
  InType invalid_input = std::make_tuple(2, std::vector<double>{1.0, 2.0, 2.0, 4.0}, std::vector<double>{3.0, 6.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_mpi_, MatrixWithoutDiagonalDominance) {  // NOLINT
  InType invalid_input = std::make_tuple(
      5, std::vector<double>{10.0, -1.0, -2.0, -3.0, -4.0, -10.0, 5.0,  -1.0, -2.0, -3.0, -4.0, -5.0, 8.0,
                             -1.0, -2.0, -3.0, -4.0, -5.0, 12.0,  -1.0, -2.0, -3.0, -4.0, -5.0, 20.0},
      std::vector<double>{0.0, -11.0, -4.0, -1.0, 6.0});
  {
    PapulinaYSimpleIterationSEQ task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_mpi_, ZeroOnDiagonal) {  // NOLINT
  InType invalid_input = std::make_tuple(3, std::vector<double>{5.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 4.0},
                                         std::vector<double>{8.0, 3.0, 6.0});
  {
    PapulinaYSimpleIterationMPI task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }
  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
TEST(PapulinaYSimpleIterationValidation_mpi_, ExplicitZeroOnDiagonal) {  // NOLINT
  InType invalid_input = std::make_tuple(3, std::vector<double>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                         std::vector<double>{8.0, 2.0, 4.0});

  {
    PapulinaYSimpleIterationMPI task(invalid_input);
    EXPECT_FALSE(task.Validation());
  }

  EXPECT_TRUE(ppc::util::DestructorFailureFlag::Get());
  ppc::util::DestructorFailureFlag::Unset();
}
}  // namespace papulina_y_simple_iteration
