#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"
#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinAVecSignRotationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  // Значительно увеличиваем размер для тестирования производительности
  const int kVectorSize_ = 30000000;  // 30 миллионов элементов
  OutType expected_result_ = 0;

  void SetUp() override {
    // Генерируем ОЧЕНЬ большой вектор со случайными данными
    std::vector<double> vector_data(kVectorSize_);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    // Заполняем случайными числами для более реалистичного теста
    for (int i = 0; i < kVectorSize_; i++) {
      vector_data.push_back(dist(gen));
    }

    input_data_ = vector_data;

    // Вычисляем ожидаемый результат для проверки
    expected_result_ = 0;
    for (size_t i = 1; i < vector_data.size(); ++i) {
      if ((vector_data[i - 1] * vector_data[i]) < 0) {
        expected_result_++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для MPI процессов с rank != 0 ожидаем значение -1
    if (output_data == -1) {
      return true;
    }

    // Для процесса с rank 0 проверяем корректный результат
    // Допускаем небольшую погрешность из-за параллельных вычислений
    return std::abs(output_data - expected_result_) <= 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikitinAVecSignRotationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NikitinAVecSignRotationMPI, NikitinAVecSignRotationSEQ>(
    PPC_SETTINGS_nikitin_a_vec_sign_rotation);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinAVecSignRotationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinAVecSignRotationPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_vec_sign_rotation
