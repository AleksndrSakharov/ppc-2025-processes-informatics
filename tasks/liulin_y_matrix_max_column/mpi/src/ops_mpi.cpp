#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"

namespace liulin_y_matrix_max_column {

int LiulinYMatrixMaxColumnMPI::TournamentMax(const std::vector<int> &column) {
  if (column.empty()) {
    return std::numeric_limits<int>::min();
  }

  int size = static_cast<int>(column.size());
  std::vector<int> temp = column;

  while (size > 1) {
    int new_size = 0;
    for (int i = 0; i < size; i += 2) {
      temp[new_size] = (i + 1 < size) ? std::max(temp[i], temp[i + 1]) : temp[i];
      ++new_size;
    }
    size = new_size;
  }
  return temp[0];
}

LiulinYMatrixMaxColumnMPI::LiulinYMatrixMaxColumnMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool LiulinYMatrixMaxColumnMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.empty() || in[0].empty()) {
    return true;
  }

  const std::size_t cols = in[0].size();

  for (const auto &row : in) {
    if (row.size() != cols) {
      return true;
    }
  }

  return true;
}

bool LiulinYMatrixMaxColumnMPI::PreProcessingImpl() {
  return true;
}

namespace {
void PrepareCounts(int rows, int cols, int world_size, std::vector<int> &sendcounts, std::vector<int> &displs) {
  sendcounts.assign(world_size, 0);
  displs.assign(world_size, 0);

  int base_cols = cols / world_size;
  int remainder = cols % world_size;

  for (int proc = 0; proc < world_size; ++proc) {
    int local_cols = base_cols + (proc < remainder ? 1 : 0);
    sendcounts[proc] = local_cols * rows;
    if (proc > 0) {
      displs[proc] = displs[proc - 1] + sendcounts[proc - 1];
    }
  }
}

void FillFlatMatrix(const InType &in, int rows, int cols, std::vector<int> &flat_matrix) {
  flat_matrix.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      flat_matrix[(col * rows) + row] = in[row][col];
    }
  }
}

std::vector<int> ComputeLocalMaxes(const std::vector<int> &local_data, int rows, int local_cols) {
  std::vector<int> local_maxes(local_cols, std::numeric_limits<int>::min());
  for (int col_idx = 0; col_idx < local_cols; ++col_idx) {
    std::vector<int> column(rows);
    for (int row = 0; row < rows; ++row) {
      column[row] = local_data[(col_idx * rows) + row];
    }
    local_maxes[col_idx] = LiulinYMatrixMaxColumnMPI::TournamentMax(column);
  }
  return local_maxes;
}
}  // namespace

bool LiulinYMatrixMaxColumnMPI::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int rows = 0;
  int cols = 0;
  if (world_rank == 0 && !in.empty() && !in[0].empty()) {
    rows = static_cast<int>(in.size());
    cols = static_cast<int>(in[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows <= 0 || cols <= 0) {
    out.clear();
    return true;
  }

  out.assign(cols, std::numeric_limits<int>::min());

  std::vector<int> sendcounts;
  std::vector<int> displs;
  PrepareCounts(rows, cols, world_size, sendcounts, displs);

  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    FillFlatMatrix(in, rows, cols, flat_matrix);
  }

  int local_cols = sendcounts[world_rank] / rows;
  int local_elements = local_cols * rows;
  std::vector<int> local_data(local_elements);

  MPI_Scatterv(world_rank == 0 ? flat_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               local_data.data(), local_elements, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_maxes = ComputeLocalMaxes(local_data, rows, local_cols);

  std::vector<int> recvcounts(world_size);
  std::vector<int> displs_gather(world_size, 0);
  for (int proc = 0; proc < world_size; ++proc) {
    recvcounts[proc] = sendcounts[proc] / rows;
    if (proc > 0) {
      displs_gather[proc] = displs_gather[proc - 1] + recvcounts[proc - 1];
    }
  }

  MPI_Gatherv(local_maxes.data(), local_cols, MPI_INT, out.data(), recvcounts.data(), displs_gather.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Bcast(out.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool LiulinYMatrixMaxColumnMPI::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_matrix_max_column
