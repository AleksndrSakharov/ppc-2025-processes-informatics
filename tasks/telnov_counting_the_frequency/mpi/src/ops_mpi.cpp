#include "telnov_counting_the_frequency/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "telnov_counting_the_frequency/common/include/common.hpp"

namespace telnov_counting_the_frequency {

TelnovCountingTheFrequencyMPI::TelnovCountingTheFrequencyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovCountingTheFrequencyMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool TelnovCountingTheFrequencyMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TelnovCountingTheFrequencyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  uint64_t n = 0;
  if (rank == 0) {
    n = static_cast<uint64_t>(GlobalData::g_data_string.size());
  }
  // Броадкаст размера (используем фиксированный тип uint64_t)
  MPI_Bcast(&n, 1, MPI_UINT64_T, /*root=*/0, MPI_COMM_WORLD);

  // Подготовим буфер на всех рангах
  if (rank != 0) {
    GlobalData::g_data_string.clear();
    GlobalData::g_data_string.resize(static_cast<size_t>(n), '\0');
  }

  if (n > 0) {
    // Передаём данные
    MPI_Bcast(const_cast<char *>(GlobalData::g_data_string.data()), static_cast<int>(n), MPI_CHAR, /*root=*/0,
              MPI_COMM_WORLD);
  }

  // Теперь все ранги имеют одинаковую g_data_string
  const std::string &s = GlobalData::g_data_string;
  size_t total_length = s.size();

  // Разбиение работы между ранги
  size_t chunk = total_length / static_cast<size_t>(size);
  size_t start = static_cast<size_t>(rank) * chunk;
  size_t end = (rank == size - 1 ? total_length : start + chunk);

  int64_t local = 0;
  for (size_t i = start; i < end; i++) {
    if (s[i] == 'X') {
      local++;
    }
  }

  int64_t total = 0;

  // Сложим локальные счётчики
  MPI_Allreduce(&local, &total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<int>(total);

  using Clock = std::chrono::high_resolution_clock;
  auto delay_start = Clock::now();
  while (std::chrono::duration<double>(Clock::now() - delay_start).count() < 0.001) {
  }

  return true;
}

bool TelnovCountingTheFrequencyMPI::PostProcessingImpl() {
  return GetOutput() == GetInput();
}

}  // namespace telnov_counting_the_frequency
