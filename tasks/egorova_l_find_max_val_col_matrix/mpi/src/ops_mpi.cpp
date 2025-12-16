#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"

namespace egorova_l_find_max_val_col_matrix {

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

EgorovaLFindMaxValColMatrixMPI::EgorovaLFindMaxValColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(0);
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

bool EgorovaLFindMaxValColMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return true;
  }

  if (matrix[0].empty()) {
    return true;
  }

  const std::size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool EgorovaLFindMaxValColMatrixMPI::PreProcessingImpl() {
  return true;
}

bool EgorovaLFindMaxValColMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  return RunMPIAlgorithm();
}

bool EgorovaLFindMaxValColMatrixMPI::RunMPIAlgorithm() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = static_cast<int>(GetInput()[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределение столбцов по процессам
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;
  int start_col = 0;
  int local_cols_count = 0;

  if (rank < remainder) {
    start_col = rank * (cols_per_proc + 1);
    local_cols_count = cols_per_proc + 1;
  } else {
    start_col = (remainder * (cols_per_proc + 1)) + ((rank - remainder) * cols_per_proc);
    local_cols_count = cols_per_proc;
  }

  // Получение локальной части матрицы
  std::vector<int> local_matrix_part = GetLocalMatrixPart(rank, size, rows, cols, start_col, local_cols_count);
  std::vector<int> local_max = CalculateLocalMaxima(local_matrix_part, rows, local_cols_count);
  std::vector<int> all_max = GatherResults(local_max, size, cols);

  GetOutput() = all_max;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::GetLocalMatrixPart(int rank, int size, int rows, int cols,
                                                                    int start_col, int local_cols_count) {
  std::vector<int> local_part(static_cast<std::size_t>(rows) * static_cast<std::size_t>(local_cols_count));

  if (rank == 0) {
    const auto &matrix = GetInput();

    // Процесс 0 заполняет свою локальную часть
    for (int ii = 0; ii < rows; ++ii) {
      for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
        const int global_col = start_col + local_idx;
        local_part[(static_cast<std::size_t>(ii) * static_cast<std::size_t>(local_cols_count)) + local_idx] =
            matrix[ii][global_col];
      }
    }

    // Отправка частей матрицы другим процессам
    SendMatrixPartsToOtherRanks(size, rows, cols);
  } else {
    // Получение данных от процесса 0
    MPI_Recv(local_part.data(), static_cast<int>(local_part.size()), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return local_part;
}

void EgorovaLFindMaxValColMatrixMPI::SendMatrixPartsToOtherRanks(int size, int rows, int cols) {
  for (int dest_rank = 1; dest_rank < size; ++dest_rank) {
    std::vector<int> dest_part = PrepareMatrixPartForRank(dest_rank, size, rows, cols);

    // Отправка данных процессу-получателю
    MPI_Send(dest_part.data(), static_cast<int>(dest_part.size()), MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
  }
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::PrepareMatrixPartForRank(int dest_rank, int size, int rows, int cols) {
  // Вычисление диапазона столбцов для процесса-получателя
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;

  int dest_start_col = 0;
  int dest_cols_count = 0;

  if (dest_rank < remainder) {
    dest_start_col = dest_rank * (cols_per_proc + 1);
    dest_cols_count = cols_per_proc + 1;
  } else {
    dest_start_col = (remainder * (cols_per_proc + 1)) + ((dest_rank - remainder) * cols_per_proc);
    dest_cols_count = cols_per_proc;
  }

  // Подготовка данных для отправки
  const auto &matrix = GetInput();
  std::vector<int> dest_part(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dest_cols_count));

  for (int ii = 0; ii < rows; ++ii) {
    for (int jj = 0; jj < dest_cols_count; ++jj) {
      const int global_col = dest_start_col + jj;
      dest_part[(static_cast<std::size_t>(ii) * static_cast<std::size_t>(dest_cols_count)) + jj] =
          matrix[ii][global_col];
    }
  }

  return dest_part;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::CalculateLocalMaxima(const std::vector<int> &local_matrix_part,
                                                                      int rows, int local_cols_count) {
  std::vector<int> local_max(local_cols_count, std::numeric_limits<int>::min());

  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    for (int row = 0; row < rows; ++row) {
      const int value = local_matrix_part[(row * local_cols_count) + local_idx];
      local_max[local_idx] = std::max(value, local_max[local_idx]);
    }
  }

  return local_max;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::GatherResults(const std::vector<int> &local_max, int size, int cols) {
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;

  std::vector<int> all_max(cols, std::numeric_limits<int>::min());
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  for (int ii = 0; ii < size; ++ii) {
    recv_counts[ii] = (ii < remainder) ? (cols_per_proc + 1) : cols_per_proc;
    displs[ii] = (ii == 0) ? 0 : displs[ii - 1] + recv_counts[ii - 1];
  }

  MPI_Allgatherv(local_max.data(), static_cast<int>(local_max.size()), MPI_INT, all_max.data(), recv_counts.data(),
                 displs.data(), MPI_INT, MPI_COMM_WORLD);

  return all_max;
}

bool EgorovaLFindMaxValColMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_find_max_val_col_matrix
