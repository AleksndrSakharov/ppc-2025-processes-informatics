#include "chyokotov_min_val_by_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"

namespace chyokotov_min_val_by_columns {

ChyokotovMinValByColumnsMPI::ChyokotovMinValByColumnsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool ChyokotovMinValByColumnsMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  size_t length_row = GetInput()[0].size();
  return std::ranges::all_of(GetInput(), [length_row](const auto &row) { return row.size() == length_row; });
}

bool ChyokotovMinValByColumnsMPI::PreProcessingImpl() {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int cols = 0;

  if (rank == 0) {
    const auto &input_matrix = GetInput();
    if (!input_matrix.empty() && !input_matrix[0].empty()) {
      size_t rows = input_matrix.size();
      cols = static_cast<int>(input_matrix[0].size());

      std::vector<std::vector<int>> tmp(cols, std::vector<int>(rows));
      for (size_t i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          tmp[j][i] = input_matrix[i][j];
        }
      }

      GetInput() = tmp;
    }
  }

  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().resize(cols, INT_MAX);

  return true;
}
namespace {
void DistributeColumns(int rank, int size, int rows, int cols, int recv_size, int loc_start,
                       std::vector<std::vector<int>> &local_cols, const std::vector<std::vector<int>> &matrix) {
  if (rank == 0) {
    for (int i = 0; i < recv_size; i++) {
      local_cols[i] = matrix[loc_start + i];
    }

    for (int dest = 1; dest < size; dest++) {
      int dest_base = cols / size;
      int dest_rem = cols % size;
      int dest_size = dest_base + (dest < dest_rem ? 1 : 0);
      int dest_start = (dest * dest_base) + std::min(dest, dest_rem);

      for (int i = 0; i < dest_size; i++) {
        MPI_Send(matrix[dest_start + i].data(), rows, MPI_INT, dest, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    for (int i = 0; i < recv_size; i++) {
      MPI_Recv(local_cols[i].data(), rows, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

std::vector<int> CalculateLocalMins(const std::vector<std::vector<int>> &local_cols, int recv_size) {
  std::vector<int> local_mins(recv_size, INT_MAX);
  for (int i = 0; i < recv_size; i++) {
    for (size_t j = 0; j < local_cols[i].size(); j++) {
      local_mins[i] = std::min(local_mins[i], local_cols[i][j]);
    }
  }
  return local_mins;
}

void PrepareGathervParams(int cols, int size, std::vector<int> &counts, std::vector<int> &displs) {
  int base = cols / size;
  int rem = cols % size;
  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i * base) + std::min(i, rem);
  }
}

}  // namespace

bool ChyokotovMinValByColumnsMPI::RunImpl() {
  int rank{};
  int size{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows{};
  int cols{};

  if (rank == 0) {
    const auto &matrix = GetInput();
    if (matrix.empty()) {
      rows = 0;
      cols = 0;
    } else {
      rows = static_cast<int>(matrix[0].size());
      cols = static_cast<int>(matrix.size());
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    return true;
  }

  int base = cols / size;
  int rem = cols % size;
  int recv_size = base + (rank < rem ? 1 : 0);
  int loc_start = (rank * base) + std::min(rank, rem);

  std::vector<std::vector<int>> local_cols(recv_size, std::vector<int>(rows));

  const auto &matrix = GetInput();
  DistributeColumns(rank, size, rows, cols, recv_size, loc_start, local_cols, matrix);

  std::vector<int> local_mins = CalculateLocalMins(local_cols, recv_size);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  PrepareGathervParams(cols, size, counts, displs);

  MPI_Allgatherv(local_mins.data(), recv_size, MPI_INT, GetOutput().data(), counts.data(), displs.data(), MPI_INT,
                 MPI_COMM_WORLD);
  return true;
}

bool ChyokotovMinValByColumnsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_min_val_by_columns
