#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>

#include "../../common/include/common.hpp"
#include "../../mpi/include/trapezoid_integration_mpi.hpp"
#include "../../seq/include/trapezoid_integration_sequential.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_v_trapezoid_seq {

TEST(KuterginTrapezoidFunc, SeqVersionWorks) {
  ASSERT_DOUBLE_EQ(kutergin_v_trapezoid_seq::Func(0.0), 0.0);
  ASSERT_DOUBLE_EQ(kutergin_v_trapezoid_seq::Func(2.0), 4.0);
  ASSERT_DOUBLE_EQ(kutergin_v_trapezoid_seq::Func(-3.0), 9.0);
}

TEST(KuterginTrapezoidFunc, MpiVersionWorks) {
  ASSERT_DOUBLE_EQ(kutergin_v_trapezoid_mpi::Func(0.0), 0.0);
  ASSERT_DOUBLE_EQ(kutergin_v_trapezoid_mpi::Func(2.0), 4.0);
}

TEST(KuterginTrapezoidValidation, SeqFailsOnInvalidBounds) {
  InType invalid_input{1.0, 0.0, 100};
  TrapezoidIntegrationSequential task(invalid_input);
  EXPECT_FALSE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(KuterginTrapezoidValidation, SeqFailsOnZeroSteps) {
  InType invalid_input{0.0, 1.0, 0};
  TrapezoidIntegrationSequential task(invalid_input);
  EXPECT_FALSE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(KuterginTrapezoidValidation, SeqFailsOnNegativeSteps) {
  InType invalid_input{0.0, 1.0, -100};
  TrapezoidIntegrationSequential task(invalid_input);
  EXPECT_FALSE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(KuterginTrapezoidValidation, MPIFailsOnNegativeSteps) {
  InType invalid_input{0.0, 1.0, -100};
  kutergin_v_trapezoid_mpi::TrapezoidIntegrationMPI task(invalid_input);
  EXPECT_FALSE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

class KuterginVRunFuncTestsSEQ
    : public ppc::util::BaseRunFuncTests<InType, OutType,
                                         TestType>  // наследник ppc::util::BaseRunFuncTests<InType, OutType, TestType>
{
 public:
  static std::string PrintTestParam(const TestType &test_param)  // вывод имен тестов
  {
    return std::get<2>(test_param);
  }

 protected:
  // подготовка входных данных перед каждым тестом
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(
        GetParam());                         // получение параметров для текущего теста
    input_data_ = std::get<0>(params);       // извлечение входных данных
    expected_output_ = std::get<1>(params);  // извлечение ожидаемого результата
  }

  // проверка результата после выполнения задачи
  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0)  // проверку выполняет только нулевой процесс
    {
      std::cout << "-> RANK 0 EXPECTED: " << expected_output_ << '\n';
      std::cout << "-> RANK 0 ACTUAL:   " << output_data << '\n';
      return std::abs(output_data - expected_output_) < 1e-6;
    }
    return true;  // у всех остальных процессов все в порядке
  }

  // возврат подготовленных входных данных для задачи
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_{};
};

namespace  // анонимное пространство имен
{

TEST_P(KuterginVRunFuncTestsSEQ, TrapezoidTest)  // параметризованный тест
{
  ExecuteTest(GetParam());
}

// массив с наборами тестовых данных
const std::array<TestType, 9> kTestCases = {
    // Успешные тесты
    std::make_tuple(InputData{.a = 0.0, .b = 3.0, .n = 10000}, 9.0, "f_x_squared_0_to_3_n_10000"),
    std::make_tuple(InputData{.a = -1.0, .b = 1.0, .n = 20000}, 0.666666, "f_x_squared_neg1_to_1_n_20000"),
    std::make_tuple(InputData{.a = 5.0, .b = 10.0, .n = 30000}, 291.666666, "f_x_squared_5_to_10_n_30000"),

    // Тесты на крайние случаи
    std::make_tuple(InputData{.a = 0.0, .b = 0.0, .n = 1000}, 0.0, "edge_zero_length_interval"),
    std::make_tuple(InputData{.a = 99.999, .b = 100.0, .n = 1000}, 9.9999, "edge_very_small_interval"),
    std::make_tuple(InputData{.a = 0.0, .b = 1.0, .n = 1}, 0.5, "edge_single_trapezoid"),

    // Тесты с "неудобным" n для проверки распределения остатка
    std::make_tuple(InputData{.a = 0.0, .b = 1.0, .n = 1001}, 0.333333, "remainder_check_n_is_odd"),
    std::make_tuple(InputData{.a = 0.0, .b = 2.0, .n = 50}, 2.6672, "remainder_check_small_n"),
    std::make_tuple(InputData{.a = 0.0, .b = 3.0, .n = 99}, 9.000459, "another_remainder_check")};

// используем фреймворк для подготовки задач к запуску
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TrapezoidIntegrationSequential, InType>(
                       kTestCases, PPC_SETTINGS_kutergin_v_trapezoid_method_of_integration),
                   ppc::util::AddFuncTask<kutergin_v_trapezoid_mpi::TrapezoidIntegrationMPI, InType>(
                       kTestCases, PPC_SETTINGS_kutergin_v_trapezoid_method_of_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KuterginVRunFuncTestsSEQ::PrintFuncTestName<KuterginVRunFuncTestsSEQ>;

// "регистрация" набора тестов и параметров в GTest
// NOLINTNEXTLINE(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables)
INSTANTIATE_TEST_SUITE_P(TrapezoidIntegrationSEQ, KuterginVRunFuncTestsSEQ, kGtestValues, kTestName);
}  // namespace

}  // namespace kutergin_v_trapezoid_seq
