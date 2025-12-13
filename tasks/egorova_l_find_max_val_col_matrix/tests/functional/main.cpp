#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"
#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_type = std::get<0>(test_params);

    switch (test_type) {
      case 0:  // Пустая матрица
        input_data_ = {};
        break;

      case 1:  // Нулевая матрица (1x0)
        input_data_ = {{}};
        break;

      case 2:  // Один столбец
        input_data_ = {{5}, {-3}, {10}, {1}};
        break;

      case 3:  // Одна строка
        input_data_ = {{8, -2, 15, 0, -7}};
        break;

      case 4:  // Квадратная матрица 3x3
        input_data_ = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        break;

      case 5:  // Неквадратная матрица (2x4)
        input_data_ = {{10, 20, 30, 40}, {50, 60, 70, 80}};
        break;

      case 6:  // Матрица с отрицательными значениями
        input_data_ = {{-5, -2, -10}, {-1, -8, -3}, {-7, -4, -6}};
        break;

      case 7:  // Матрица с одинаковыми значениями
        input_data_ = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
        break;

      case 8:  // Матрица с максимальным элементом в разных столбцах
        input_data_ = {{1, 100, 2}, {50, 3, 200}, {10, 4, 5}};
        break;

      case 9:  // Большая матрица 5x5
        input_data_ = {
            {15, 8, 22, 4, 19}, {7, 25, 11, 3, 14}, {9, 2, 30, 17, 6}, {12, 5, 1, 28, 10}, {20, 13, 16, 21, 0}};
        break;

      case 10:  // Матрица с одним элементом
        input_data_ = {{42}};
        break;

      case 11:  // Матрица где максимум в первом столбце
        input_data_ = {{99, 1, 2}, {88, 3, 4}, {77, 5, 6}};
        break;

      case 12:  // Матрица где максимум в последнем столбце
        input_data_ = {{1, 2, 100}, {3, 4, 90}, {5, 6, 80}};
        break;

      case 13:  // Матрица с нулевыми значениями
        input_data_ = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        break;

      case 14:  // Одна строка большая
        input_data_ = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
        break;

      case 15:  // Один столбец большой
        input_data_ = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
        break;

      case 16:  // Смешанные значения
        input_data_ = {{-1, 5, -3}, {2, -8, 10}, {7, 0, -5}};
        break;

      case 17:  // Матрица для неравномерного распределения столбцов
        input_data_ = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
        break;

      default:
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = GetTestInputData();

    // Для пустых матриц ожидаем пустой результат
    if (matrix.empty() || matrix[0].empty()) {
      return output_data.empty();
    }

    // Для обычных матриц проверяем корректность результата
    if (output_data.size() != matrix[0].size()) {
      return false;
    }

    std::vector<int> expected(matrix[0].size(), std::numeric_limits<int>::min());
    for (std::size_t jj = 0; jj < matrix[0].size(); ++jj) {
      for (std::size_t ii = 0; ii < matrix.size(); ++ii) {
        expected[jj] = std::max(matrix[ii][jj], expected[jj]);
      }
    }

    for (std::size_t ii = 0; ii < output_data.size(); ++ii) {
      if (output_data[ii] != expected[ii]) {
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

TEST_P(EgorovaLRunFuncTestsProcesses, FindMaxValColMatrix) {
  ExecuteTest(GetParam());
}

// тип теста, описание
const std::array<TestType, 18> kTestParam = {std::make_tuple(0, "empty_matrix"),
                                             std::make_tuple(1, "zero_matrix"),
                                             std::make_tuple(2, "single_column"),
                                             std::make_tuple(3, "single_row"),
                                             std::make_tuple(4, "square_3x3"),
                                             std::make_tuple(5, "non_square_2x4"),
                                             std::make_tuple(6, "negative_values"),
                                             std::make_tuple(7, "same_values"),
                                             std::make_tuple(8, "max_in_different_cols"),
                                             std::make_tuple(9, "large_5x5"),
                                             std::make_tuple(10, "single_element"),
                                             std::make_tuple(11, "max_in_first_col"),
                                             std::make_tuple(12, "max_in_last_col"),
                                             std::make_tuple(13, "zero_values"),
                                             std::make_tuple(14, "single_row_large"),
                                             std::make_tuple(15, "single_column_large"),
                                             std::make_tuple(16, "mixed_positive_negative"),
                                             std::make_tuple(17, "uneven_column_distribution")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<EgorovaLFindMaxValColMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_egorova_l_find_max_val_col_matrix),
                                           ppc::util::AddFuncTask<EgorovaLFindMaxValColMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_egorova_l_find_max_val_col_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = EgorovaLRunFuncTestsProcesses::PrintFuncTestName<EgorovaLRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, EgorovaLRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace egorova_l_find_max_val_col_matrix
