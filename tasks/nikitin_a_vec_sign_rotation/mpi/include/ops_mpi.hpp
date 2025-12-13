#pragma once

#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinAVecSignRotationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikitinAVecSignRotationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Функция для главного процесса (rank 0) - распределяет данные и собирает результаты
  static int MainProcess(int process_count, const std::vector<double> &data);

  // Функция для рабочих процессов (rank 1, 2, ...) - получает данные и вычисляет свою часть
  static void WorkerProcess();

  // Вспомогательная функция для проверки смены знака между двумя числами
  static bool IsSignChange(double first_value, double second_value);
};

}  // namespace nikitin_a_vec_sign_rotation
