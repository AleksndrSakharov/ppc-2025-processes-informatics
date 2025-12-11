#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // для std::max
#include <cstddef>    // для size_t
#include <limits>     // для std::numeric_limits
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"

namespace pylaeva_s_max_elem_matrix {

PylaevaSMaxElemMatrixMPI::PylaevaSMaxElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<int>::min();
}

bool PylaevaSMaxElemMatrixMPI::ValidationImpl() {
  const auto rows = static_cast<size_t>(std::get<0>(GetInput()));
  const auto columns = static_cast<size_t>(std::get<1>(GetInput()));
  size_t matrix_size = rows * columns;

  return (matrix_size == std::get<2>(GetInput()).size()) && (rows > 0) && (columns > 0);
}

bool PylaevaSMaxElemMatrixMPI::PreProcessingImpl() {
  return true;
}

bool PylaevaSMaxElemMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t matrix_size = 0;
  std::vector<int> matrix_data;

  if (rank == 0) {
    const auto matrix_rows = static_cast<size_t>(std::get<0>(GetInput()));
    const auto matrix_columns = static_cast<size_t>(std::get<1>(GetInput()));
    matrix_size = matrix_rows * matrix_columns;
    matrix_data = std::get<2>(GetInput());
  }

  MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  int local_size = static_cast<int>(matrix_size / size);
  int remainder = static_cast<int>(matrix_size % size);

  int offset = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = local_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += sendcounts[i];
  }

  int local_elements = sendcounts[rank];
  std::vector<int> local_data(local_elements);

  MPI_Scatterv(rank == 0 ? matrix_data.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_data.data(),
               local_elements, MPI_INT, 0, MPI_COMM_WORLD);

  int local_max = std::numeric_limits<int>::min();
  for (int i = 0; i < local_elements; i++) {
    local_max = std::max(local_max, local_data[i]);
  }

  int global_max = std::numeric_limits<int>::min();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool PylaevaSMaxElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
