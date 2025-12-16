#include "sannikov_i_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"

namespace sannikov_i_column_sum {

SannikovIColumnSumMPI::SannikovIColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIColumnSumMPI::ValidationImpl() {
  const auto &input_matrix = GetInput();
  if (input_matrix.empty() || input_matrix.front().empty()) {
    return false;
  }

  const std::size_t columns = input_matrix.front().size();
  for (const auto &row : input_matrix) {
    if (row.size() != columns) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIColumnSumMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void SannikovIColumnSumMPI::PrepareSendBuffer(const InType &input_matrix, int rank, std::uint64_t rows,
                                              std::uint64_t columns, std::vector<int> &sendbuf) {
  if (rank != 0) {
    return;
  }
  if (rank == 0) {
    const std::uint64_t base = rows * columns;
    sendbuf.resize(static_cast<std::size_t>(base));
    for (std::uint64_t i = 0; i < rows; i++) {
      for (std::uint64_t j = 0; j < columns; j++) {
        sendbuf[static_cast<std::size_t>((i * columns) + (j))] =
            input_matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      }
    }
  }
}

bool SannikovIColumnSumMPI::RunImpl() {
  const auto &input_matrix = GetInput();

  int rank = 0;
  int size = 1;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    rows = static_cast<std::uint64_t>(input_matrix.size());
    columns = static_cast<std::uint64_t>(input_matrix.front().size());
  }

  MPI_Bcast(&rows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  const std::uint64_t base = rows * columns;
  if (columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      (base > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))) {
    return false;
  }
  const int columns_int = static_cast<int>(columns);
  const int base_int = static_cast<int>(base);
  GetOutput().assign(static_cast<std::size_t>(columns_int), 0);

  std::vector<int> sendbuf;
  PrepareSendBuffer(input_matrix, rank, rows, columns, sendbuf);
  std::vector<int> elem_for_proc(size);
  std::vector<int> id_elem(size);
  int displacement = 0;
  for (int i = 0; i < size; i++) {
    elem_for_proc[i] = static_cast<int>(base_int / size) + (i < (base_int % size) ? 1 : 0);
    id_elem[i] = displacement;
    displacement += elem_for_proc[i];
  }
  const int mpi_displacement = id_elem[rank] % static_cast<int>(columns_int);
  std::vector<int> buf(static_cast<std::size_t>(elem_for_proc[rank]), 0);
  MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, elem_for_proc.data(), id_elem.data(), MPI_INT, buf.data(),
               elem_for_proc[rank], MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> sum(static_cast<std::size_t>(columns_int), 0);
  for (int i = 0; i < (elem_for_proc[rank]); i++) {
    int new_col = (i + mpi_displacement) % columns_int;
    sum[static_cast<std::size_t>(new_col)] += buf[static_cast<std::size_t>(i)];
  }
  MPI_Allreduce(sum.data(), GetOutput().data(), columns_int, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return !GetOutput().empty();
}

bool SannikovIColumnSumMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_column_sum
