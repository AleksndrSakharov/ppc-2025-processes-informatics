#include "kosolapov_v_max_values_in_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixMPI::KosolapovVMaxValuesInColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &matrix = GetInput();
    if (matrix.empty()) {
      return false;
    }
    for (size_t i = 0; i < matrix.size() - 1; i++) {
      if ((matrix[i].size() != matrix[i + 1].size()) || (matrix[i].empty())) {
        return false;
      }
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &matrix = GetInput();
    if (!matrix.empty() && !matrix[0].empty()) {
      GetOutput().resize(matrix[0].size());
    }
  }
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::RunImpl() {
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::vector<int>> local_matrix;
  int rows = 0;
  int columns = 0;
  if (rank == 0) {
    const auto &matrix = GetInput();
    rows = static_cast<int>(matrix.size());
    columns = static_cast<int>(matrix[0].size());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int rows_per_proc = rows / processes_count;
  const int remainder = rows % processes_count;
  const int start = (rank * rows_per_proc) + std::min(rank, remainder);
  const int end = start + rows_per_proc + (rank < remainder ? 1 : 0);
  const int local_rows = end - start;

  if (rank == 0) {
    DistributeDataFromRoot(local_matrix, start, local_rows, columns, processes_count, rows_per_proc, remainder);
  } else {
    local_matrix.resize(local_rows);
    for (int i = 0; i < local_rows; i++) {
      local_matrix[i].resize(columns);
      MPI_Recv(local_matrix[i].data(), columns, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  auto local_maxs = CalculateLocalMax(local_matrix, columns);
  std::vector<int> global_maxs(columns);
  MPI_Allreduce(local_maxs.data(), global_maxs.data(), columns, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_maxs;
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::PostProcessingImpl() {
  return true;
}

void KosolapovVMaxValuesInColMatrixMPI::DistributeDataFromRoot(std::vector<std::vector<int>> &local_matrix, int start,
                                                               int local_rows, int columns, int processes_count,
                                                               int rows_per_proc, int remainder) {
  const auto &matrix = GetInput();
  local_matrix.resize(local_rows);

  for (int i = 0; i < local_rows; i++) {
    local_matrix[i] = matrix[start + i];
  }

  for (int proc = 1; proc < processes_count; proc++) {
    const int proc_start = (proc * rows_per_proc) + std::min(proc, remainder);
    const int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
    const int proc_rows_count = proc_end - proc_start;

    for (int i = 0; i < proc_rows_count; i++) {
      MPI_Send(matrix[proc_start + i].data(), columns, MPI_INT, proc, i, MPI_COMM_WORLD);
    }
  }
}
std::vector<int> KosolapovVMaxValuesInColMatrixMPI::CalculateLocalMax(const std::vector<std::vector<int>> &matrix,
                                                                      const int columns) {
  std::vector<int> local_maxs(columns, std::numeric_limits<int>::min());
  for (const auto &row : matrix) {
    for (int i = 0; i < columns; i++) {
      local_maxs[i] = std::max(row[i], local_maxs[i]);
    }
  }
  return local_maxs;
}
}  // namespace kosolapov_v_max_values_in_col_matrix
