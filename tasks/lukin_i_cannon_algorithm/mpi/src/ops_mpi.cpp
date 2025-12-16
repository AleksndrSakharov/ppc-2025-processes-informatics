#include "lukin_i_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"

namespace lukin_i_cannon_algorithm {

LukinICannonAlgorithmMPI::LukinICannonAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = OutType();
}

bool LukinICannonAlgorithmMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int proc_count = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    int grid_size = static_cast<int>(std::floor(std::sqrt(proc_count)));
    int rsize_a = static_cast<int>(std::get<0>(GetInput()).size());
    int rsize_b = static_cast<int>(std::get<1>(GetInput()).size());
    size_ = std::get<2>(GetInput());
    return (rsize_a > 0) && (rsize_b > 0) && (rsize_a == size_ * size_) && (rsize_a == rsize_b) &&
           (rsize_a % grid_size == 0);
  }
  return true;
}

bool LukinICannonAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool LukinICannonAlgorithmMPI::RunImpl() {
  int global_rank = -1;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  MPI_Bcast(&size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // случай, если 1 на 1 решетка процессов - просто seq версия
  if (proc_count < 4) {
    return RunSeq(global_rank);
  }

  // для процессов, напрямую учавствующих в вычислениях, создается другой коммуникатор
  return RunCannon(global_rank, proc_count);
}

bool LukinICannonAlgorithmMPI::PostProcessingImpl() {
  return true;
}

void LukinICannonAlgorithmMPI::MulNSum(const double *a, const double *b, double *c, int size) {
  for (int i = 0; i < size; i++) {
    for (int k = 0; k < size; k++) {
      double fixed = a[(i * size) + k];
      for (int j = 0; j < size; j++) {
        c[(i * size) + j] += fixed * b[(k * size) + j];
      }
    }
  }
}

bool LukinICannonAlgorithmMPI::RunSeq(int global_rank) {
  std::vector<double> c(static_cast<size_t>(size_ * size_));
  if (global_rank == 0) {
    double *a = std::get<0>(GetInput()).data();
    double *b = std::get<1>(GetInput()).data();
    LukinICannonAlgorithmMPI::MulNSum(a, b, c.data(), size_);
  }
  MPI_Bcast(c.data(), size_ * size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(c);
  return true;
}

bool LukinICannonAlgorithmMPI::RunCannon(int global_rank, int proc_count) {
  int grid_size = static_cast<int>(std::floor(std::sqrt(proc_count)));
  int working_proc_count = grid_size * grid_size;
  MPI_Comm mpi_comm_cannon = MPI_COMM_NULL;
  int color = (global_rank < working_proc_count) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, global_rank, &mpi_comm_cannon);

  if (mpi_comm_cannon != MPI_COMM_NULL) {
    CannonWorkers(mpi_comm_cannon, grid_size, working_proc_count);
    MPI_Comm_free(&mpi_comm_cannon);
  } else {
    CannonNonWorkers();
  }

  return true;
}

void LukinICannonAlgorithmMPI::CannonWorkers(MPI_Comm comm, int grid_size, int working_proc_count) {
  int cannon_rank = -1;
  MPI_Comm_rank(comm, &cannon_rank);

  int block_size = size_ / grid_size;
  int block_elems = block_size * block_size;
  std::vector<double> a_block(block_elems);
  std::vector<double> b_block(block_elems);
  std::vector<double> c_block(block_elems, 0);

  std::vector<double> a_blocks;
  std::vector<double> b_blocks;

  // ручная упаковка
  if (cannon_rank == 0) {
    a_blocks.resize(static_cast<size_t>(working_proc_count) * static_cast<size_t>(block_elems));
    b_blocks.resize(static_cast<size_t>(working_proc_count) * static_cast<size_t>(block_elems));

    MatrixPack(a_blocks.data(), b_blocks.data(), working_proc_count, block_elems, block_size, grid_size);
  }

  MPI_Scatter(a_blocks.data(), block_elems, MPI_DOUBLE, a_block.data(), block_elems, MPI_DOUBLE, 0, comm);

  MPI_Scatter(b_blocks.data(), block_elems, MPI_DOUBLE, b_block.data(), block_elems, MPI_DOUBLE, 0, comm);

  int row = cannon_rank / grid_size;
  int col = cannon_rank % grid_size;

  // начальный сдвиг
  int left = (row * grid_size) + ((col - row + grid_size) % grid_size);
  int right = (row * grid_size) + ((col + row) % grid_size);

  MPI_Sendrecv_replace(a_block.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, comm, MPI_STATUS_IGNORE);

  int up = (((row - col + grid_size) % grid_size) * grid_size) + col;
  int down = (((row + col) % grid_size) * grid_size) + col;

  MPI_Sendrecv_replace(b_block.data(), block_elems, MPI_DOUBLE, up, 0, down, 0, comm, MPI_STATUS_IGNORE);

  // цикл умножения и сдвига
  for (int iter = 0; iter < grid_size; iter++) {
    LukinICannonAlgorithmMPI::MulNSum(a_block.data(), b_block.data(), c_block.data(), block_size);

    if (iter < grid_size - 1) {
      left = (row * grid_size) + ((col - 1 + grid_size) % grid_size);
      right = (row * grid_size) + ((col + 1) % grid_size);

      MPI_Sendrecv_replace(a_block.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, comm, MPI_STATUS_IGNORE);

      up = (((row - 1 + grid_size) % grid_size) * grid_size) + col;
      down = (((row + 1) % grid_size) * grid_size) + col;

      MPI_Sendrecv_replace(b_block.data(), block_elems, MPI_DOUBLE, up, 0, down, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  // упаковка данных в результирующую
  std::vector<double> c_blocks(static_cast<size_t>(size_ * size_));
  MPI_Gather(c_block.data(), block_elems, MPI_DOUBLE, c_blocks.data(), block_elems, MPI_DOUBLE, 0, comm);

  std::vector<double> c(static_cast<size_t>(size_ * size_));
  if (cannon_rank == 0) {
    MatrixUnpack(c.data(), c_blocks.data(), working_proc_count, block_elems, block_size, grid_size);
  }

  MPI_Bcast(c.data(), size_ * size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(c);
}

void LukinICannonAlgorithmMPI::CannonNonWorkers() {
  std::vector<double> c(static_cast<size_t>(size_ * size_));
  MPI_Bcast(c.data(), size_ * size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(c);
}

void LukinICannonAlgorithmMPI::MatrixPack(double *a_blocks, double *b_blocks, int working_proc_count, int block_elems,
                                          int block_size, int grid_size) {
  double *a = std::get<0>(GetInput()).data();
  double *b = std::get<1>(GetInput()).data();

  for (int proc = 0; proc < working_proc_count; proc++) {
    int proc_i = proc / grid_size;
    int proc_j = proc % grid_size;
    int buf_offset = proc * block_elems;

    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        int global_i = (proc_i * block_size) + i;
        int global_j = (proc_j * block_size) + j;
        int global_idx = (global_i * size_) + global_j;
        int buf_idx = buf_offset + (i * block_size) + j;

        a_blocks[buf_idx] = a[global_idx];
        b_blocks[buf_idx] = b[global_idx];
      }
    }
  }
}

void LukinICannonAlgorithmMPI::MatrixUnpack(double *c, const double *c_blocks, int working_proc_count, int block_elems,
                                            int block_size, int grid_size) const {
  for (int proc = 0; proc < working_proc_count; proc++) {
    int proc_i = proc / grid_size;
    int proc_j = proc % grid_size;
    int buf_offset = proc * block_elems;

    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        int global_i = (proc_i * block_size) + i;
        int global_j = (proc_j * block_size) + j;
        int global_idx = (global_i * size_) + global_j;
        int buf_idx = buf_offset + (i * block_size) + j;

        c[global_idx] = c_blocks[buf_idx];
      }
    }
  }
}

}  // namespace lukin_i_cannon_algorithm
