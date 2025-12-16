#include "kondakov_v_min_val_in_matrix_str/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"

namespace kondakov_v_min_val_in_matrix_str {

KondakovVMinValMatrixMPI::KondakovVMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp = in;
  GetInput().swap(tmp);
  GetOutput().clear();
  GetOutput().resize(in.size());
}

bool KondakovVMinValMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return true;
  }

  size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return !row.empty() && row.size() == cols; });
}

bool KondakovVMinValMatrixMPI::PreProcessingImpl() {
  return true;
}

bool KondakovVMinValMatrixMPI::RunImpl() {
  int n = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto [total_rows, cols] = BroadcastMatrixDimensions();
  if (total_rows == 0) {
    GetOutput().clear();
    return true;
  }

  int local_row_count = static_cast<int>(total_rows / n) + (std::cmp_less(rank, total_rows % n) ? 1 : 0);
  if (static_cast<uint64_t>(local_row_count) * cols > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Local matrix slice too large (exceeds INT_MAX elements)");
  }

  auto local_flat = ScatterMatrixData(total_rows, cols, n, rank);
  auto local_minima = ComputeLocalMinima(local_flat, cols, local_row_count);
  GetOutput() = GatherMinima(local_minima, total_rows, n, rank);

  return true;
}

std::pair<size_t, size_t> KondakovVMinValMatrixMPI::BroadcastMatrixDimensions() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t total_rows = 0;
  size_t cols = 0;
  if (rank == 0) {
    const auto &in_data = GetInput();
    total_rows = in_data.size();
    cols = (total_rows > 0) ? in_data[0].size() : 0;
  }

  uint64_t total_rows_u64 = total_rows;
  uint64_t cols_u64 = cols;
  MPI_Bcast(&total_rows_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  return {static_cast<size_t>(total_rows_u64), static_cast<size_t>(cols_u64)};
}

std::vector<int> KondakovVMinValMatrixMPI::ScatterMatrixData(size_t total_rows, size_t cols, int n, int rank) {
  std::vector<int> send_counts(n, 0);
  std::vector<int> send_displs(n, 0);

  PrepareSendInfo(rank, total_rows, cols, n, send_counts, send_displs);

  MPI_Bcast(send_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(send_displs.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  int local_row_count = static_cast<int>(total_rows / n);
  if (std::cmp_less(rank, total_rows % n)) {
    ++local_row_count;
  }

  std::vector<int> local_flat(local_row_count * cols);

  if (rank == 0) {
    const auto &in_data = GetInput();
    std::vector<int> flat_data(total_rows * cols);
    size_t idx = 0;
    for (const auto &row : in_data) {
      std::ranges::copy(row, flat_data.begin() + static_cast<std::vector<int>::difference_type>(idx));
      idx += row.size();
    }
    assert(idx == total_rows * cols);

    MPI_Scatterv(flat_data.data(), send_counts.data(), send_displs.data(), MPI_INT, local_flat.data(),
                 static_cast<int>(local_flat.size()), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, send_counts.data(), send_displs.data(), MPI_INT, local_flat.data(),
                 static_cast<int>(local_flat.size()), MPI_INT, 0, MPI_COMM_WORLD);
  }

  return local_flat;
}

void KondakovVMinValMatrixMPI::PrepareSendInfo(int rank, size_t total_rows, size_t cols, int n,
                                               std::vector<int> &send_counts, std::vector<int> &send_displs) {
  if (rank != 0) {
    return;
  }

  int extra = static_cast<int>(total_rows % n);
  for (int i = 0; i < n; ++i) {
    int rows_i = static_cast<int>(total_rows / n);
    if (i < extra) {
      ++rows_i;
    }
    uint64_t count = static_cast<uint64_t>(rows_i) * cols;
    if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Matrix too large for MPI (slice exceeds INT_MAX)");
    }
    send_counts[i] = static_cast<int>(count);
  }

  for (int i = 1; i < n; ++i) {
    send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
  }
}

std::vector<int> KondakovVMinValMatrixMPI::ComputeLocalMinima(const std::vector<int> &local_flat, size_t cols,
                                                              int local_row_count) {
  std::vector<int> local_minima(local_row_count);
  for (int i = 0; i < local_row_count; ++i) {
    int min_val = std::numeric_limits<int>::max();
    const int *row = local_flat.data() + (i * cols);
    for (size_t j = 0; j < cols; ++j) {
      min_val = std::min(min_val, row[j]);
    }
    local_minima[i] = min_val;
  }
  return local_minima;
}

std::vector<int> KondakovVMinValMatrixMPI::GatherMinima(const std::vector<int> &local_minima, size_t total_rows, int n,
                                                        int rank) {
  std::vector<int> recv_counts(n);
  std::vector<int> displs(n);

  if (n > 0 && rank == 0) {
    int extra = static_cast<int>(total_rows % n);
    for (int i = 0; i < n; ++i) {
      recv_counts[i] = static_cast<int>(total_rows / n) + ((i < extra) ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < n; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  MPI_Bcast(recv_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> global_minima(total_rows);
  MPI_Gatherv(local_minima.data(), static_cast<int>(local_minima.size()), MPI_INT,
              rank == 0 ? global_minima.data() : nullptr, recv_counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Bcast(global_minima.data(), static_cast<int>(total_rows), MPI_INT, 0, MPI_COMM_WORLD);
  return global_minima;
}

bool KondakovVMinValMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_min_val_in_matrix_str
