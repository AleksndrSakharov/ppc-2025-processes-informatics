#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

namespace {

void LocalMultiply(const std::vector<double> &a_block, const std::vector<double> &b_block, std::vector<double> &c_block,
                   int local_rows, int k_dim, int local_cols) {
  for (int i = 0; i < local_rows; ++i) {
    for (int p = 0; p < k_dim; ++p) {
      double a_val = a_block[static_cast<std::size_t>(i) * static_cast<std::size_t>(k_dim) + p];
      for (int j = 0; j < local_cols; ++j) {
        c_block[static_cast<std::size_t>(i) * static_cast<std::size_t>(local_cols) + j] +=
            a_val * b_block[static_cast<std::size_t>(p) * static_cast<std::size_t>(local_cols) + j];
      }
    }
  }
}

}  // namespace

SakharovACannonAlgorithmMPI::SakharovACannonAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmMPI::ValidationImpl() {
  return IsValidInput(GetInput());
}

bool SakharovACannonAlgorithmMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  auto out_size = static_cast<std::size_t>(input.rows_a) * static_cast<std::size_t>(input.cols_b);
  GetOutput().assign(out_size, 0.0);
  return true;
}

bool SakharovACannonAlgorithmMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  const int m = input.rows_a;
  const int k = input.cols_a;
  const int n = input.cols_b;

  if (world_size == 1) {
    auto &output = GetOutput();
    for (int i = 0; i < m; ++i) {
      for (int p = 0; p < k; ++p) {
        double a_val = input.a[Idx(k, i, p)];
        for (int j = 0; j < n; ++j) {
          output[Idx(n, i, j)] += a_val * input.b[Idx(n, p, j)];
        }
      }
    }
    return true;
  }

  int base_rows = m / world_size;
  int extra_rows = m % world_size;

  std::vector<int> row_counts(world_size);
  std::vector<int> row_displs(world_size);
  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    row_counts[i] = base_rows + (i < extra_rows ? 1 : 0);
    row_displs[i] = offset;
    offset += row_counts[i];
  }

  int local_rows = row_counts[rank];

  std::vector<double> local_a(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(k));
  std::vector<double> local_c(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n), 0.0);

  std::vector<int> send_counts_a(world_size);
  std::vector<int> displs_a(world_size);
  for (int i = 0; i < world_size; ++i) {
    send_counts_a[i] = row_counts[i] * k;
    displs_a[i] = row_displs[i] * k;
  }

  MPI_Scatterv(input.a.data(), send_counts_a.data(), displs_a.data(), MPI_DOUBLE, local_a.data(), local_rows * k,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> b_data;
  if (rank == 0) {
    b_data = input.b;
  } else {
    b_data.resize(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
  }
  MPI_Bcast(b_data.data(), k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  LocalMultiply(local_a, b_data, local_c, local_rows, k, n);

  std::vector<int> recv_counts_c(world_size);
  std::vector<int> displs_c(world_size);
  for (int i = 0; i < world_size; ++i) {
    recv_counts_c[i] = row_counts[i] * n;
    displs_c[i] = row_displs[i] * n;
  }

  MPI_Gatherv(local_c.data(), local_rows * n, MPI_DOUBLE, GetOutput().data(), recv_counts_c.data(), displs_c.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(GetOutput().data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool SakharovACannonAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
