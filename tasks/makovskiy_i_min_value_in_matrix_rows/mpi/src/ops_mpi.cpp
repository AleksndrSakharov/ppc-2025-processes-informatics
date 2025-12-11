#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

namespace {
void SendDataToWorkers(const InType &matrix, int size, int rows_per_proc, int remaining_rows, int &current_row_idx) {
  for (int i = 1; i < size; ++i) {
    const int rows_for_this_proc = rows_per_proc + (i < remaining_rows ? 1 : 0);
    MPI_Send(&rows_for_this_proc, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    for (int j = 0; j < rows_for_this_proc; ++j) {
      const auto &row = matrix[current_row_idx++];
      const auto row_size = static_cast<int>(row.size());
      MPI_Send(&row_size, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
      if (row_size > 0) {
        MPI_Send(row.data(), row_size, MPI_INT, i, 2, MPI_COMM_WORLD);
      }
    }
  }
}
}  // namespace

void MinValueMPI::ProcessRankZero(std::vector<int> &local_min_values) {
  const auto &matrix = this->GetInput();
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 0) {
    return;
  }

  const auto num_rows = static_cast<int>(matrix.size());
  const int rows_per_proc = num_rows / size;
  const int remaining_rows = num_rows % size;
  int current_row_idx = 0;

  const int rows_for_root = rows_per_proc + (0 < remaining_rows ? 1 : 0);
  for (int j = 0; j < rows_for_root; ++j) {
    const auto &row = matrix[current_row_idx++];
    if (!row.empty()) {
      local_min_values.push_back(*std::ranges::min_element(row));
    }
  }

  if (size > 1) {
    SendDataToWorkers(matrix, size, rows_per_proc, remaining_rows, current_row_idx);
  }
}

void MinValueMPI::ProcessWorkerRank(std::vector<int> &local_min_values) {
  int num_local_rows = 0;
  MPI_Recv(&num_local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (int i = 0; i < num_local_rows; ++i) {
    int row_size = 0;
    MPI_Recv(&row_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (row_size > 0) {
      std::vector<int> received_row(row_size);
      MPI_Recv(received_row.data(), row_size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_min_values.push_back(*std::ranges::min_element(received_row));
    }
  }
}

void MinValueMPI::GatherResults(const std::vector<int> &local_min_values) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto local_results_count = static_cast<int>(local_min_values.size());
  std::vector<int> recvcounts;
  if (rank == 0) {
    recvcounts.resize(size);
  }

  MPI_Gather(&local_results_count, 1, MPI_INT, (rank == 0 ? recvcounts.data() : nullptr), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<int> displs;
  if (rank == 0 && size > 0) {
    displs.resize(size, 0);
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  }

  MPI_Gatherv(local_min_values.data(), local_results_count, MPI_INT, (rank == 0 ? this->GetOutput().data() : nullptr),
              (rank == 0 ? recvcounts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_INT, 0,
              MPI_COMM_WORLD);
}

MinValueMPI::MinValueMPI(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool MinValueMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 0;
  if (rank == 0) {
    const auto &mat = this->GetInput();
    is_valid = !mat.empty() ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool MinValueMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &mat = this->GetInput();
    this->GetOutput().clear();
    this->GetOutput().resize(mat.size());
  }
  return true;
}

bool MinValueMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> local_min_values;
  if (rank == 0) {
    ProcessRankZero(local_min_values);
  } else {
    ProcessWorkerRank(local_min_values);
  }

  GatherResults(local_min_values);

  return true;
}

bool MinValueMPI::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_min_value_in_matrix_rows
