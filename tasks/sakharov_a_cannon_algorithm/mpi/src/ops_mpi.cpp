#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
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

  for (int row_idx = 0; row_idx < block_size; ++row_idx) {
    const int global_row = row_offset + row_idx;
    const std::size_t src_offset = Offset(matrix_size, global_row, col_offset);
    const std::ptrdiff_t dst_offset = static_cast<std::ptrdiff_t>(row_idx) * block_size;
    std::copy_n(src.begin() + static_cast<std::ptrdiff_t>(src_offset), block_size, dst.begin() + dst_offset);
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

  for (int row_idx = 0; row_idx < block_size; ++row_idx) {
    const std::size_t dst_offset = Offset(matrix_size, start_row + row_idx, start_col);
    const std::ptrdiff_t src_offset = static_cast<std::ptrdiff_t>(row_idx) * block_size;
    std::copy_n(block.begin() + src_offset, block_size, dst.begin() + static_cast<std::ptrdiff_t>(dst_offset));
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

bool RunSingleProcess(const InType &input, OutType &output) {
  const int block_size = SelectBlockSize(input.size);
  output = BlockMultiply(input, block_size);
  return true;
}

MPI_Comm CreateSubCommunicator(bool participate, int rank) {
  MPI_Comm sub_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, participate ? 1 : MPI_UNDEFINED, rank, &sub_comm);
  return sub_comm;
}

bool BroadcastSequentialForNonParticipants(bool participate, int rank, const InType &input, OutType &output) {
  if (participate) {
    return false;
  }

  if (rank == 0) {
    RunSingleProcess(input, output);
  }
  MPI_Bcast(output.data(), static_cast<int>(output.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

MPI_Comm CreateCartesian(MPI_Comm sub_comm, int grid_dim) {
  MPI_Comm cart_comm = MPI_COMM_NULL;
  const std::array<int, 2> dims{grid_dim, grid_dim};
  const std::array<int, 2> periods{1, 1};
  MPI_Cart_create(sub_comm, 2, dims.data(), periods.data(), 1, &cart_comm);
  return cart_comm;
}

void BuildScatterBuffers(const InType &input, int n, int grid_dim, int block_size, int block_elems,
                         std::vector<double> &send_a, std::vector<double> &send_b) {
  const int sub_size = grid_dim * grid_dim;
  send_a.resize(static_cast<std::size_t>(sub_size) * static_cast<std::size_t>(block_elems));
  send_b.resize(static_cast<std::size_t>(sub_size) * static_cast<std::size_t>(block_elems));

  for (int row_idx = 0; row_idx < grid_dim; ++row_idx) {
    for (int col_idx = 0; col_idx < grid_dim; ++col_idx) {
      std::vector<double> tmp_a;
      std::vector<double> tmp_b;
      CopyBlock(input.a, n, block_size, row_idx, col_idx, tmp_a);
      CopyBlock(input.b, n, block_size, row_idx, col_idx, tmp_b);

      const int dest = (row_idx * grid_dim) + col_idx;
      const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(dest) * block_elems;
      std::copy_n(tmp_a.begin(), block_elems, send_a.begin() + offset);
      std::copy_n(tmp_b.begin(), block_elems, send_b.begin() + offset);
    }
  }
}

void AssembleBlocks(const std::vector<double> &gathered, int block_elems, int block_size, int grid_dim, int n,
                    OutType &output) {
  const int sub_size = grid_dim * grid_dim;
  for (int idx = 0; idx < sub_size; ++idx) {
    const int row_idx = idx / grid_dim;
    const int col_idx = idx % grid_dim;
    const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(idx) * block_elems;
    std::vector<double> tmp(block_elems);
    std::copy_n(gathered.begin() + offset, block_elems, tmp.begin());
    PlaceBlock(tmp, block_size, row_idx, col_idx, n, output);
  }
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
    return RunSingleProcess(GetInput(), GetOutput());
  }

  const int grid_dim = ComputeGridDim(world_size);
  const int active = grid_dim * grid_dim;
  const bool participate = (grid_dim > 0) && (n % grid_dim == 0) && rank < active;

  MPI_Comm sub_comm = CreateSubCommunicator(participate, rank);
  if (BroadcastSequentialForNonParticipants(participate, rank, GetInput(), GetOutput())) {
    return true;
  }

  int sub_rank = 0;
  int sub_size = 0;
  MPI_Comm_rank(sub_comm, &sub_rank);
  MPI_Comm_size(sub_comm, &sub_size);

  MPI_Comm cart_comm = CreateCartesian(sub_comm, grid_dim);

  std::array<int, 2> coords{0, 0};
  MPI_Cart_coords(cart_comm, sub_rank, 2, coords.data());
  const int block_size = n / grid_dim;
  const int block_elems = block_size * block_size;

  std::vector<double> a_block(static_cast<std::size_t>(block_elems));
  std::vector<double> b_block(static_cast<std::size_t>(block_elems));
  std::vector<double> c_block(static_cast<std::size_t>(block_elems), 0.0);

  std::vector<double> send_a;
  std::vector<double> send_b;
  if (sub_rank == 0) {
    BuildScatterBuffers(GetInput(), n, grid_dim, block_size, block_elems, send_a, send_b);
  }

  MPI_Scatter(send_a.data(), block_elems, MPI_DOUBLE, a_block.data(), block_elems, MPI_DOUBLE, 0, sub_comm);
  MPI_Scatter(send_b.data(), block_elems, MPI_DOUBLE, b_block.data(), block_elems, MPI_DOUBLE, 0, sub_comm);

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
    AssembleBlocks(gathered, block_elems, block_size, grid_dim, n, GetOutput());
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
