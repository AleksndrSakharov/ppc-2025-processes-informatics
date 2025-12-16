#include "rychkova_d_sum_matrix_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "rychkova_d_sum_matrix_columns/common/include/common.hpp"

namespace rychkova_d_sum_matrix_columns {

RychkovaDSumMatrixColumnsMPI::RychkovaDSumMatrixColumnsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput()[i] = in[i];
  }
  GetOutput() = OutType{};
}

bool RychkovaDSumMatrixColumnsMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = true;

  if (rank == 0) {
    const auto &input = GetInput();

    if (!input.empty()) {
      size_t cols = input[0].size();
      for (const auto &row : input) {
        if (row.size() != cols) {
          is_valid = false;
          break;
        }
      }
    }

    is_valid = is_valid && GetOutput().empty();
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

  return is_valid;
}

bool RychkovaDSumMatrixColumnsMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool has_data = false;
  size_t num_cols = 0;

  if (rank == 0) {
    const auto &input = GetInput();
    has_data = !input.empty();
    if (has_data) {
      num_cols = input[0].size();
    }
  }

  MPI_Bcast(&has_data, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (has_data) {
    GetOutput() = std::vector<int>(num_cols, 0);
  } else {
    GetOutput() = std::vector<int>{};
  }

  return true;
}

namespace {

void CalculateDistribution(int size, size_t num_rows, size_t num_cols, std::vector<int> &send_counts,
                           std::vector<int> &displacements) {
  size_t rows_per_process = num_rows / static_cast<size_t>(size);
  size_t remainder = num_rows % static_cast<size_t>(size);

  size_t current_displacement = 0;
  for (int i = 0; i < size; ++i) {
    auto process_index = static_cast<size_t>(i);
    size_t rows_for_process_i = rows_per_process + (process_index < remainder ? 1 : 0);
    send_counts[i] = static_cast<int>(rows_for_process_i * num_cols);
    displacements[i] = static_cast<int>(current_displacement);
    current_displacement += rows_for_process_i * num_cols;
  }
}

std::vector<int> FlattenMatrix(const InType &matrix) {
  std::vector<int> flat_matrix;
  if (!matrix.empty()) {
    size_t num_rows = matrix.size();
    size_t num_cols = matrix[0].size();
    flat_matrix.reserve(num_rows * num_cols);
    for (const auto &row : matrix) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  }
  return flat_matrix;
}

std::vector<int> ComputeLocalSums(const std::vector<int> &local_data, size_t local_rows, size_t num_cols) {
  std::vector<int> local_sums(num_cols, 0);
  for (size_t i = 0; i < local_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      local_sums[j] += local_data[(i * num_cols) + j];
    }
  }
  return local_sums;
}

}  // namespace

bool RychkovaDSumMatrixColumnsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &output = GetOutput();
  bool has_data = !output.empty();
  size_t num_cols = output.size();

  if (!has_data) {
    std::vector<int> empty_sums(0);
    MPI_Allreduce(MPI_IN_PLACE, empty_sums.data(), 0, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return true;
  }

  size_t num_rows = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    num_rows = input.size();
  }
  MPI_Bcast(&num_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    CalculateDistribution(size, num_rows, num_cols, send_counts, displacements);
  }

  MPI_Bcast(send_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  size_t rows_per_process = num_rows / static_cast<size_t>(size);
  size_t remainder = num_rows % static_cast<size_t>(size);
  size_t local_rows = rows_per_process + (std::cmp_less(rank, remainder) ? 1 : 0);

  std::vector<int> flat_matrix;
  if (rank == 0) {
    flat_matrix = FlattenMatrix(GetInput());
  }

  std::vector<int> local_data(local_rows * num_cols);
  MPI_Scatterv(rank == 0 ? flat_matrix.data() : nullptr, send_counts.data(), displacements.data(), MPI_INT,
               local_data.data(), static_cast<int>(local_rows * num_cols), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_sums = ComputeLocalSums(local_data, local_rows, num_cols);

  MPI_Allreduce(local_sums.data(), GetOutput().data(), static_cast<int>(num_cols), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  return true;
}

bool RychkovaDSumMatrixColumnsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rychkova_d_sum_matrix_columns
