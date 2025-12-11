#include "guseva_a_matrix_sums/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "guseva_a_matrix_sums/common/include/common.hpp"

namespace guseva_a_matrix_sums {

GusevaAMatrixSumsMPI::GusevaAMatrixSumsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GusevaAMatrixSumsMPI::ValidationImpl() {
  return (static_cast<uint64_t>(std::get<0>(GetInput())) * std::get<1>(GetInput()) == std::get<2>(GetInput()).size()) &&
         (GetOutput().empty());
}

bool GusevaAMatrixSumsMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool GusevaAMatrixSumsMPI::RunImpl() {
  int rank = 0;
  int wsize = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &wsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = static_cast<int>(std::get<0>(GetInput()));
  int columns = static_cast<int>(std::get<1>(GetInput()));

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int size = rows * columns;

  int elems_per_proc = size / wsize;
  int remainder = size % wsize;

  std::vector<int> counts(wsize, 0);
  std::vector<int> displs(wsize, 0);

  int displ = 0;
  int shift = 0;
  int total = 0;

  for (int proc = 0; proc < wsize; proc++) {
    counts[proc] = elems_per_proc + (proc < remainder ? 1 : 0);
    displs[proc] = displ;
    displ += counts[proc];
    if (rank == proc) {
      shift = total % columns;
    }
    total += counts[proc];
  }

  std::vector<double> matrix(size, 0);

  if (rank == 0) {
    matrix = std::get<2>(GetInput());
  }

  std::vector<double> local_buff(counts[rank], 0);

  MPI_Scatterv(matrix.data(), counts.data(), displs.data(), MPI_DOUBLE, local_buff.data(), counts[rank], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> local_sums(columns, 0.0);

  for (int i = 0; i < counts[rank]; i++) {
    local_sums[(i + shift) % columns] += local_buff[i];
  }

  if (rank == 0) {
    GetOutput().resize(columns, 0.0);
  }

  MPI_Reduce(local_sums.data(), GetOutput().data(), columns, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool GusevaAMatrixSumsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace guseva_a_matrix_sums
