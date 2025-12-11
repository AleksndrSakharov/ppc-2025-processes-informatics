#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"
#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace spichek_d_dot_product_of_vectors {

class SpichekDDotProductOfVectorsRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[vectors, description] = test_param;
    return description;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);  // Получаем пару векторов
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[vector1, vector2] = input_data_;

    // Вычисляем ожидаемый результат
    int expected_result = 0;
    for (size_t i = 0; i < vector1.size(); ++i) {
      expected_result += vector1[i] * vector2[i];
    }

    return (output_data == expected_result);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SpichekDDotProductOfVectorsRunFuncTestsProcesses, DotProductTest) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: ((вектор1, вектор2), описание)
const std::array<TestType, 17> kTestParam = {
    // 1. Простые
    std::make_tuple(std::make_pair(std::vector<int>{1, 2, 3}, 
                                   std::vector<int>{4, 5, 6}), 
                    "basic_vectors_3"),

    // 2. Нули
    std::make_tuple(std::make_pair(std::vector<int>{0, 0, 0}, 
                                   std::vector<int>{1, 2, 3}), 
                    "all_zero_first"),

    std::make_tuple(std::make_pair(std::vector<int>{1, 2, 3}, 
                                   std::vector<int>{0, 0, 0}), 
                    "all_zero_second"),

    // 3. Только отрицательные
    std::make_tuple(std::make_pair(std::vector<int>{-1, -2, -3},
                                   std::vector<int>{-4, -5, -6}),
                    "all_negative"),

    // 4. Смешанные
    std::make_tuple(std::make_pair(std::vector<int>{1, -2, 3, -4},
                                   std::vector<int>{-1, 2, -3, 4}),
                    "alternating_signs"),

    // 5. Большие числа (проверка переполнения)
    std::make_tuple(std::make_pair(std::vector<int>{10000, 20000, 30000},
                                   std::vector<int>{30000, 20000, 10000}),
                    "large_numbers"),

    // 6. Короткий случай
    std::make_tuple(std::make_pair(std::vector<int>{7},
                                   std::vector<int>{9}),
                    "single_element"),

    // 7. Много одинаковых
    std::make_tuple(std::make_pair(std::vector<int>(100, 5),
                                   std::vector<int>(100, 2)),
                    "constant_100"),

    // 8. Вектор длиной 1000
    std::make_tuple(std::make_pair(std::vector<int>(1000, 1),
                                   std::vector<int>(1000, -1)),
                    "size_1000"),

    // 9. Рандом-1 (детерминированный)
    std::make_tuple(std::make_pair(
                        []{
                            std::vector<int> v(50);
                            for (int i = 0; i < 50; i++){ v[i] = i - 25;
}
                            return v;
                        }(),
                        []{
                            std::vector<int> v(50);
                            for (int i = 0; i < 50; i++){ v[i] = 25 - i;
}
                            return v;
                        }()),
                    "random_like_case_1"),

    // 10. Рандом-2 (другая закономерность)
    std::make_tuple(std::make_pair(
                        []{
                            std::vector<int> v(60);
                            for (int i = 0; i < 60; i++){ v[i] = ((i * 7) % 13) - 6;
}
                            return v;
                        }(),
                        []{
                            std::vector<int> v(60);
                            for (int i = 0; i < 60; i++){ v[i] = ((i * 3) % 11) - 5;
}
                            return v;
                        }()),
                    "random_like_case_2"),

    // 11. Половина нулей
    std::make_tuple(std::make_pair(std::vector<int>{0,1,0,1,0,1},
                                   std::vector<int>{1,0,1,0,1,0}),
                    "half_zero"),

    // 12. Восходящая/нисходящая последовательность
    std::make_tuple(std::make_pair(
                        []{
                            std::vector<int> v(100);
                            for (int i = 0; i < 100; i++){ v[i] = i;
}
                            return v;
                        }(),
                        []{
                            std::vector<int> v(100);
                            for (int i = 0; i < 100; i++){ v[i] = 100 - i;
}
                            return v;
                        }()),
                    "ascending_descending"),

    // 13. Большой тест (5000 элементов)
    std::make_tuple(std::make_pair(std::vector<int>(5000, 3),
                                   std::vector<int>(5000, 4)),
                    "size_5000"),

    // 14. Очень большие положительные/отрицательные
    std::make_tuple(std::make_pair(std::vector<int>{1000000, -1000000},
                                   std::vector<int>{-500000, 500000}),
                    "extreme_values"),

    // 15. Векторы одинаковых случайных чисел
    std::make_tuple(std::make_pair(std::vector<int>(200, 123),
                                   std::vector<int>(200, 321)),
                    "same_random_like"),
    
    // 16. Пустые векторы
    std::make_tuple(std::make_pair(std::vector<int>{}, 
                                   std::vector<int>{}),
                      "empty_vectors")
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SpichekDDotProductOfVectorsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_spichek_d_dot_product_of_vectors),
                                           ppc::util::AddFuncTask<SpichekDDotProductOfVectorsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_spichek_d_dot_product_of_vectors));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SpichekDDotProductOfVectorsRunFuncTestsProcesses::PrintFuncTestName<
    SpichekDDotProductOfVectorsRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(DotProductTests, SpichekDDotProductOfVectorsRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace spichek_d_dot_product_of_vectors
