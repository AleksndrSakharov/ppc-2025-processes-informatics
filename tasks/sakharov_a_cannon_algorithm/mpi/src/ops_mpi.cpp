#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"

namespace sakharov_a_cannon_algorithm {

namespace {

int ComputeGridDim(int world_size) {
  const int dim = static_cast<int>(std::sqrt(static_cast<double>(world_size)));
  return dim;
}

void CopyBlock(const std::vector<double> &src, int matrix_size, int block_size, int block_row, int block_col,
               std::vector<double> &dst) {
  dst.assign(static_cast<std::size_t>(block_size) * static_cast<std::size_t>(block_size), 0.0);

  const int row_offset = block_row * block_size;
  const int col_offset = block_col * block_size;

  for (int r = 0; r < block_size; ++r) {
    const int global_row = row_offset + r;
    const std::size_t src_offset = Offset(matrix_size, global_row, col_offset);
    std::copy_n(src.begin() + static_cast<std::ptrdiff_t>(src_offset), block_size,
                dst.begin() + static_cast<std::ptrdiff_t>(r * block_size));
  }
}

void MultiplyLocal(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    const int row_offset = i * block_size;
    for (int k = 0; k < block_size; ++k) {
      const double a_val = a[row_offset + k];
      const int b_row_offset = k * block_size;
      for (int j = 0; j < block_size; ++j) {
        c[row_offset + j] += a_val * b[b_row_offset + j];
      }
    }
  }
}

void PlaceBlock(const std::vector<double> &block, int block_size, int block_row, int block_col, int matrix_size,
                std::vector<double> &dst) {
  const int start_row = block_row * block_size;
  const int start_col = block_col * block_size;

  for (int r = 0; r < block_size; ++r) {
    const std::size_t dst_offset = Offset(matrix_size, start_row + r, start_col);
    std::copy_n(block.begin() + static_cast<std::ptrdiff_t>(r * block_size), block_size,
                dst.begin() + static_cast<std::ptrdiff_t>(dst_offset));
  }
}

void ShiftLeft(MPI_Comm cart_comm, std::vector<double> &block, int block_elems) {
  int src_left = MPI_PROC_NULL;
  int dst_left = MPI_PROC_NULL;
  MPI_Cart_shift(cart_comm, 1, -1, &src_left, &dst_left);
  MPI_Sendrecv_replace(block.data(), block_elems, MPI_DOUBLE, dst_left, 0, src_left, 0, cart_comm, MPI_STATUS_IGNORE);
}

void ShiftUp(MPI_Comm cart_comm, std::vector<double> &block, int block_elems) {
  int src_up = MPI_PROC_NULL;
  int dst_up = MPI_PROC_NULL;
  MPI_Cart_shift(cart_comm, 0, -1, &src_up, &dst_up);
  MPI_Sendrecv_replace(block.data(), block_elems, MPI_DOUBLE, dst_up, 0, src_up, 0, cart_comm, MPI_STATUS_IGNORE);
}

void ShiftLeftN(MPI_Comm cart_comm, std::vector<double> &block, int block_elems, int steps) {
  if (steps == 0) {
    return;
  }
  int src = MPI_PROC_NULL;
  int dst = MPI_PROC_NULL;
  MPI_Cart_shift(cart_comm, 1, -steps, &src, &dst);
  MPI_Sendrecv_replace(block.data(), block_elems, MPI_DOUBLE, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);
}

void ShiftUpN(MPI_Comm cart_comm, std::vector<double> &block, int block_elems, int steps) {
  if (steps == 0) {
    return;
  }
  int src = MPI_PROC_NULL;
  int dst = MPI_PROC_NULL;
  MPI_Cart_shift(cart_comm, 0, -steps, &src, &dst);
  MPI_Sendrecv_replace(block.data(), block_elems, MPI_DOUBLE, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);
}

}  // namespace

SakharovACannonAlgorithmMPI::SakharovACannonAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovACannonAlgorithmMPI::ValidationImpl() {
  return HasValidShape(GetInput());
}

bool SakharovACannonAlgorithmMPI::PreProcessingImpl() {
  const int n = GetInput().size;
  GetOutput().assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  return true;
}

bool SakharovACannonAlgorithmMPI::RunImpl() {
  const int n = GetInput().size;

  int world_size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (world_size == 1) {
    const int block_size = SelectBlockSize(n);
    GetOutput() = BlockMultiply(GetInput(), block_size);
    return true;
  }

  const int grid_dim = ComputeGridDim(world_size);
  const int active = grid_dim * grid_dim;
  const bool divisible = (grid_dim > 0) && (n % grid_dim == 0);
  const bool participate = divisible && rank < active;

  MPI_Comm sub_comm = MPI_COMM_NULL;
  if (participate) {
    MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &sub_comm);
  } else {
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &sub_comm);
  }

  if (!participate) {
    if (rank == 0) {
      const int block_size = SelectBlockSize(n);
      GetOutput() = BlockMultiply(GetInput(), block_size);
    }
    MPI_Bcast(GetOutput().data(), static_cast<int>(GetOutput().size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return true;
  }

  int sub_rank = 0;
  int sub_size = 0;
  MPI_Comm_rank(sub_comm, &sub_rank);
  MPI_Comm_size(sub_comm, &sub_size);

  MPI_Comm cart_comm = MPI_COMM_NULL;
  const int dims[2] = {grid_dim, grid_dim};
  const int periods[2] = {1, 1};
  MPI_Cart_create(sub_comm, 2, dims, periods, 1, &cart_comm);

  int coords[2] = {0, 0};
  MPI_Cart_coords(cart_comm, sub_rank, 2, coords);
  const int block_size = n / grid_dim;
  const int block_elems = block_size * block_size;

  std::vector<double> a_block(static_cast<std::size_t>(block_elems));
  std::vector<double> b_block(static_cast<std::size_t>(block_elems));
  std::vector<double> c_block(static_cast<std::size_t>(block_elems), 0.0);

  std::vector<double> send_a;
  std::vector<double> send_b;
  if (sub_rank == 0) {
    send_a.resize(static_cast<std::size_t>(sub_size) * static_cast<std::size_t>(block_elems));
    send_b.resize(static_cast<std::size_t>(sub_size) * static_cast<std::size_t>(block_elems));

    for (int r = 0; r < grid_dim; ++r) {
      for (int c = 0; c < grid_dim; ++c) {
        std::vector<double> tmp_a;
        std::vector<double> tmp_b;
        CopyBlock(GetInput().a, n, block_size, r, c, tmp_a);
        CopyBlock(GetInput().b, n, block_size, r, c, tmp_b);

        const int dest = r * grid_dim + c;
        auto offset = static_cast<std::size_t>(dest) * static_cast<std::size_t>(block_elems);
        std::copy(tmp_a.begin(), tmp_a.end(), send_a.begin() + static_cast<std::ptrdiff_t>(offset));
        std::copy(tmp_b.begin(), tmp_b.end(), send_b.begin() + static_cast<std::ptrdiff_t>(offset));
      }
    }
  }

  MPI_Scatter(send_a.data(), block_elems, MPI_DOUBLE, a_block.data(), block_elems, MPI_DOUBLE, 0, sub_comm);
  MPI_Scatter(send_b.data(), block_elems, MPI_DOUBLE, b_block.data(), block_elems, MPI_DOUBLE, 0, sub_comm);

  // Initial alignment: shift A left by row index and B up by column index in one hop each.
  ShiftLeftN(cart_comm, a_block, block_elems, coords[0]);
  ShiftUpN(cart_comm, b_block, block_elems, coords[1]);

  for (int step = 0; step < grid_dim; ++step) {
    MultiplyLocal(a_block, b_block, c_block, block_size);
    ShiftLeft(cart_comm, a_block, block_elems);
    ShiftUp(cart_comm, b_block, block_elems);
  }

  std::vector<double> gathered;
  if (sub_rank == 0) {
    gathered.resize(static_cast<std::size_t>(sub_size) * static_cast<std::size_t>(block_elems));
  }

  MPI_Gather(c_block.data(), block_elems, MPI_DOUBLE, gathered.data(), block_elems, MPI_DOUBLE, 0, sub_comm);

  if (sub_rank == 0) {
    for (int idx = 0; idx < sub_size; ++idx) {
      const int r = idx / grid_dim;
      const int c = idx % grid_dim;
      const auto offset = static_cast<std::size_t>(idx) * static_cast<std::size_t>(block_elems);
      std::vector<double> tmp(block_elems);
      std::copy_n(gathered.begin() + static_cast<std::ptrdiff_t>(offset), block_elems, tmp.begin());
      PlaceBlock(tmp, block_size, r, c, n, GetOutput());
    }
  }

  MPI_Bcast(GetOutput().data(), static_cast<int>(GetOutput().size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (cart_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm);
  }
  if (sub_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&sub_comm);
  }

  return true;
}

bool SakharovACannonAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_cannon_algorithm
