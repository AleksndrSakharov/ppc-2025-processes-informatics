#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"

namespace nikitin_a_vec_sign_rotation {

NikitinAVecSignRotationMPI::NikitinAVecSignRotationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NikitinAVecSignRotationMPI::ValidationImpl() {
  return true;
}

bool NikitinAVecSignRotationMPI::PreProcessingImpl() {
  return true;
}

bool NikitinAVecSignRotationMPI::RunImpl() {
  // Получаем общее количество процессов в MPI коммуникаторе
  int process_count = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  // Получаем ранг (номер) текущего процесса
  int process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  // Вектор для хранения входных данных (заполняется только процессом 0)
  std::vector<double> vector_data;

  // Инициализируем выходное значение для некорневых процессов
  if (process_rank != 0) {
    GetOutput() = -1;
  }

  // Только процесс 0 получает входные данные
  if (process_rank == 0) {
    vector_data = GetInput();
    // Корректируем количество процессов, если их больше чем элементов в векторе
    process_count = std::min<int>(process_count, static_cast<int>(vector_data.size()));
  }

  // Рассылаем актуальное количество процессов всем участникам
  MPI_Bcast(&process_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Если вектор пустой или процессов нет - завершаем работу
  if (process_count == 0) {
    return true;
  }

  // Распределяем работу между процессами
  if (process_rank == 0) {
    // Главный процесс распределяет данные и собирает результаты
    GetOutput() = MainProcess(process_count, vector_data);
  } else if (process_rank < process_count) {
    // Рабочие процессы получают данные и вычисляют свою часть
    WorkerProcess();
  }

  return true;
}

bool NikitinAVecSignRotationMPI::PostProcessingImpl() {
  return true;
}

// Аналогична методу из SEQ версии
bool NikitinAVecSignRotationMPI::IsSignChange(double first_value, double second_value) {
  const bool first_negative = first_value < 0.0;
  const bool second_non_negative = second_value >= 0.0;
  const bool first_non_negative = first_value >= 0.0;
  const bool second_negative = second_value < 0.0;

  return (first_negative && second_non_negative) || (first_non_negative && second_negative);
}

int NikitinAVecSignRotationMPI::MainProcess(int process_count, const std::vector<double> &vector_data) {
  int total_swaps = 0;
  const int data_size = static_cast<int>(vector_data.size());

  // Вычисляем базовый размер части для каждого процесса
  const int base_chunk_size = data_size / process_count;
  int remainder = data_size % process_count;  // Остаток элементов для распределения

  int start_index = 0;  // Начальный индекс для распределения данных

  // Распределяем данные между рабочими процессами (rank 1, 2, ...)
  for (int worker = 1; worker < process_count; worker++) {
    int chunk_size = base_chunk_size;

    // Распределяем остаточные элементы по одному первым процессам
    if (remainder > 0) {
      chunk_size++;
      remainder--;
    }

    // Добавляем один элемент справа для проверки чередования на границе блоков
    chunk_size++;

    // Отправляем размер части рабочему процессу
    MPI_Send(&chunk_size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);

    // Отправляем сами данные рабочему процессу
    MPI_Send(vector_data.data() + start_index, chunk_size, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);

    // Сдвигаем начальный индекс для следующего блока (минус перекрывающий элемент)
    start_index += chunk_size - 1;
  }

  // Главный процесс обрабатывает оставшуюся часть данных
  const int master_chunk_size = data_size - start_index;
  for (int i = start_index + 1; i < start_index + master_chunk_size; i++) {
    if (IsSignChange(vector_data[i - 1], vector_data[i])) {
      total_swaps++;
    }
  }

  // Собираем результаты от всех рабочих процессов
  for (int worker = 1; worker < process_count; worker++) {
    int worker_swaps = 0;
    MPI_Recv(&worker_swaps, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    total_swaps += worker_swaps;
  }

  return total_swaps;
}

void NikitinAVecSignRotationMPI::WorkerProcess() {
  int chunk_size = 0;

  // Получаем размер своей части данных от главного процесса
  MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Выделяем память для локальных данных
  std::vector<double> local_data(chunk_size);

  // Получаем свои данные от главного процесса
  MPI_Recv(local_data.data(), chunk_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Подсчитываем количество чередований знаков в своей части
  int local_swaps = 0;
  for (int i = 1; i < chunk_size; i++) {
    if (IsSignChange(local_data[i - 1], local_data[i])) {
      local_swaps++;
    }
  }

  // Отправляем результат обратно главному процессу
  MPI_Send(&local_swaps, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

}  // namespace nikitin_a_vec_sign_rotation
