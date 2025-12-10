#include "../include/trapezoid_integration_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "../../common/include/common.hpp"

namespace kutergin_v_trapezoid_mpi {

double Func(double x)  // интегрируемая функция для примера
{
  return x * x;
}

TrapezoidIntegrationMPI::TrapezoidIntegrationMPI(const kutergin_v_trapezoid_seq::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput() = 0.0;                     // инициализация выходных данных
}

bool TrapezoidIntegrationMPI::ValidationImpl() {
  return (GetInput().a <= GetInput().b) && (GetInput().n > 0);
}

bool TrapezoidIntegrationMPI::PreProcessingImpl() {
  return true;
}

bool TrapezoidIntegrationMPI::RunImpl() {
  int process_rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  kutergin_v_trapezoid_seq::InType broadcast_data;
  if (process_rank == 0) {
    broadcast_data = GetInput();
  }

  // Каждое поле рассылается отдельно
  MPI_Bcast(&broadcast_data.a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&broadcast_data.b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&broadcast_data.n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double a = broadcast_data.a;
  double b = broadcast_data.b;
  int n = broadcast_data.n;
  double h = (b - a) / n;

  const int base_n = n / process_count;     // целое часть от деления числа разбиений на число процессов
  const int remainder = n % process_count;  // остаток от деления числа разбиений на число процессов

  const int local_n = base_n + (process_rank < remainder ? 1 : 0);  // количество разбиений (трапеций) на один процесс

  int start_index = 0;
  if (process_rank < remainder) {
    start_index = process_rank * (base_n + 1);
  } else {
    start_index = (remainder * (base_n + 1)) + ((process_rank - remainder) * base_n);
  }

  double local_a = a + (start_index * h);  // начало отрезка для текущего процесса

  // локальные вычисления
  double local_sum = 0.0;
  if (local_n > 0) {
    local_sum = (Func(local_a) + Func(local_a + (local_n * h))) / 2.0;
  }

  for (int i = 1; i < local_n; ++i) {
    local_sum += Func(local_a + (i * h));
  }

  // агрегация
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (process_rank == 0) {
    GetOutput() = global_sum * h;
  }

  return true;
}

bool TrapezoidIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_trapezoid_mpi
