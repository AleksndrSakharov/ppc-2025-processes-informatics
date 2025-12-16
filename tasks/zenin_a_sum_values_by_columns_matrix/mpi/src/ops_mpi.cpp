#include "zenin_a_sum_values_by_columns_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

ZeninASumValuesByColumnsMatrixMPI::ZeninASumValuesByColumnsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninASumValuesByColumnsMatrixMPI::ValidationImpl() {
  auto &input = GetInput();
  return ((std::get<0>(input)) * std::get<1>(input) == std::get<2>(input).size() && (GetOutput().empty()));
}

bool ZeninASumValuesByColumnsMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  GetOutput().clear();
  return true;
}

void ZeninASumValuesByColumnsMatrixMPI::FillSendBuffer(const std::vector<double> &mat, std::vector<double> &sendbuf,
                                                       size_t rows, size_t cols, size_t base, size_t rest,
                                                       int world_size) {
  size_t pos = 0;
  for (int proc = 0; proc < world_size; proc++) {
    auto proc_size = static_cast<size_t>(proc);
    size_t pc_begin = (proc_size * base) + (std::cmp_less(proc_size, rest) ? proc_size : rest);
    size_t pc_end = pc_begin + (base + (std::cmp_less(proc_size, rest) ? 1 : 0));
    for (size_t col = pc_begin; col < pc_end; col++) {
      for (size_t row = 0; row < rows; row++) {
        sendbuf[pos++] = mat[(row * cols) + col];
      }
    }
  }
}

bool ZeninASumValuesByColumnsMatrixMPI::RunImpl() {
  auto rows = static_cast<size_t>(std::get<0>(GetInput()));
  auto cols = static_cast<size_t>(std::get<1>(GetInput()));
  const std::vector<double> &mat = std::get<2>(GetInput());

  std::vector<double> &global_sum = GetOutput();

  int rank = 0;
  int world_size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const size_t base = cols / static_cast<size_t>(world_size);
  const size_t rest = cols % static_cast<size_t>(world_size);

  const size_t my_cols = base + (std::cmp_less(static_cast<size_t>(rank), rest) ? 1 : 0);

  std::vector<int> sendcounts(static_cast<size_t>(world_size));
  std::vector<int> displs(static_cast<size_t>(world_size));
  if (rank == 0) {
    int offset = 0;
    for (int proc = 0; proc < world_size; proc++) {
      size_t pc = base + (std::cmp_less(static_cast<size_t>(proc), rest) ? 1 : 0);
      sendcounts[proc] = static_cast<int>(pc * rows);
      displs[proc] = offset;
      offset += sendcounts[proc];
    }
  }

  std::vector<double> sendbuf;
  if (rank == 0) {
    sendbuf.resize(rows * cols);
    FillSendBuffer(mat, sendbuf, rows, cols, base, rest, world_size);
  }
  std::vector<double> local_block(rows * my_cols);
  MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_block.data(),
               static_cast<int>(local_block.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_sum(my_cols, 0.0);
  for (size_t col_id = 0; col_id < my_cols; col_id++) {
    for (size_t row_id = 0; row_id < rows; row_id++) {
      local_sum[col_id] += local_block[(col_id * rows) + row_id];
    }
  }
  std::vector<int> recvcounts(static_cast<size_t>(world_size));
  std::vector<int> recvdispls(static_cast<size_t>(world_size));

  if (rows == 0) {
    throw std::runtime_error("Matrix has zero rows");
  }

  if (rank == 0) {
    size_t offset = 0;
    for (int proc = 0; proc < world_size; proc++) {
      recvcounts[proc] = sendcounts[proc] / static_cast<int>(rows);
      recvdispls[proc] = static_cast<int>(offset);
      offset += static_cast<size_t>(recvcounts[proc]);
    }
    global_sum.assign(cols, 0.0);
  }

  MPI_Gatherv(local_sum.data(), static_cast<int>(my_cols), MPI_DOUBLE, global_sum.data(), recvcounts.data(),
              recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  global_sum.resize(cols);
  MPI_Bcast(global_sum.data(), static_cast<int>(cols), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

bool ZeninASumValuesByColumnsMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_sum_values_by_columns_matrix
