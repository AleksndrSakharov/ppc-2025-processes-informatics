#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"
#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinAVecSignRotationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);  // возвращаем строковое описание теста
  }

 protected:
  void SetUp() override {
    auto params = GetParam();
    TestType test_params = std::get<2>(params);

    test_case_id_ = std::get<0>(test_params);
    test_description_ = std::get<1>(test_params);

    // Генерируем тестовые данные в зависимости от ID теста
    switch (test_case_id_) {
      case 1:  // Простой случай - 2 чередования
        input_data_ = {1.0, -2.0, 3.0, -4.0, 5.0};
        expected_output_ = 4;
        break;

      case 2:  // Все положительные - 0 чередований
        input_data_ = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_output_ = 0;
        break;

      case 3:  // Все отрицательные - 0 чередований
        input_data_ = {-1.0, -2.0, -3.0, -4.0, -5.0};
        expected_output_ = 0;
        break;

      case 4:  // Частые чередования
        input_data_ = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
        expected_output_ = 5;
        break;

      case 5:  // С нулями (0 считается положительным)
        input_data_ = {1.0, 0.0, -1.0, 0.0, 1.0};
        expected_output_ = 2;
        break;

      case 6:  // Один элемент - 0 чередований
        input_data_ = {5.0};
        expected_output_ = 0;
        break;

      case 7:  // Два элемента с чередованием
        input_data_ = {1.0, -1.0};
        expected_output_ = 1;
        break;

      case 8:  // Два элемента без чередования
        input_data_ = {1.0, 2.0};
        expected_output_ = 0;
        break;

      case 9:  // Большой вектор для проверки производительности
        input_data_ = GenerateAlternatingVector(100);
        expected_output_ = 99;
        break;

      case 10:  // Пустой вектор - особый случай
        input_data_ = {};
        expected_output_ = 0;
        break;

      case 11:  // Смешанные дробные числа
        input_data_ = {1.5, -2.3, 0.7, -0.1, 3.14, -2.71};
        expected_output_ = 5;
        break;

      case 12:                                       // Длинная последовательность без чередований
        input_data_ = std::vector<double>(50, 1.0);  // 50 одинаковых положительных
        expected_output_ = 0;
        break;

      case 13:  // Граничные значения
        input_data_ = {std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
                       -std::numeric_limits<double>::min(), -std::numeric_limits<double>::max()};
        expected_output_ = 1;
        break;

      case 14:  // Много чередований в середине
        input_data_ = {1.0, 2.0, -1.0, -2.0, 3.0, 4.0, -3.0, -4.0};
        expected_output_ = 3;
        break;
      case 15:
        input_data_ = GenerateAlternatingVector(1000);
        expected_output_ = 999;
        break;
      case 16:
        input_data_ = {1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0};
        expected_output_ = 3;  // 1→0, 0→-1, -1→0, 0→1, 1→0, 0→-1
        break;

      default:
        throw std::runtime_error("Unknown test case ID: " + std::to_string(test_case_id_));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для MPI процессов с rank != 0 ожидаем значение -1
    if (output_data == -1) {
      return true;
    }

    // Проверяем корректность результата для процесса с rank 0
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int test_case_id_ = 0;
  std::string test_description_;
  std::vector<double> input_data_;
  int expected_output_ = 0;

  // Генератор знакопеременного вектора
  static std::vector<double> GenerateAlternatingVector(size_t size) {
    std::vector<double> result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      result.push_back((i % 2 == 0) ? static_cast<double>(i) + 1.0 : -static_cast<double>(i) - 1.0);
    }
    return result;
  }
};

namespace {

TEST_P(NikitinAVecSignRotationFuncTests, CountSignChanges) {
  ExecuteTest(GetParam());
}

// Определяем тестовые случаи с хорошим покрытием
const std::array<TestType, 14> kTestParam = {
    std::make_tuple(1, "simple_alternating"),        // Простой случай чередований
    std::make_tuple(2, "all_positive"),              // Все положительные
    std::make_tuple(3, "all_negative"),              // Все отрицательные
    std::make_tuple(4, "frequent_alternating"),      // Частые чередования
    std::make_tuple(5, "with_zeros"),                // С нулевыми значениями
    std::make_tuple(6, "single_element"),            // Один элемент
    std::make_tuple(7, "two_elements_alternating"),  // Два элемента с чередованием
    std::make_tuple(8, "two_elements_same_sign"),    // Два элемента без чередования
    std::make_tuple(9, "large_vector"),              // Большой вектор
    std::make_tuple(10, "empty_vector"),             // Пустой вектор
    std::make_tuple(11, "fractional_numbers"),       // Дробные числа
    std::make_tuple(12, "long_same_sign"),           // Длинная последовательность одного знака
    std::make_tuple(13, "boundary_values"),          // Граничные значения
    std::make_tuple(14, "mixed_pattern")             // Смешанный паттерн
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<NikitinAVecSignRotationMPI, InType>(kTestParam, PPC_SETTINGS_nikitin_a_vec_sign_rotation),
    ppc::util::AddFuncTask<NikitinAVecSignRotationSEQ, InType>(kTestParam, PPC_SETTINGS_nikitin_a_vec_sign_rotation));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NikitinAVecSignRotationFuncTests::PrintFuncTestName<NikitinAVecSignRotationFuncTests>;

INSTANTIATE_TEST_SUITE_P(SignRotationTests, NikitinAVecSignRotationFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nikitin_a_vec_sign_rotation
