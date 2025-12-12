#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"

namespace fatehov_k_matrix_max_elem {

FatehovKMatrixMaxElemMPI::FatehovKMatrixMaxElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKMatrixMaxElemMPI::ValidationImpl() {
  auto &data = GetInput();
  return (std::get<0>(data) > 0 && std::get<0>(data) <= kMaxRows) &&
         (std::get<1>(data) > 0 && std::get<1>(data) <= kMaxCols) &&
         (std::get<0>(data) * std::get<1>(data) <= kMaxMatrixSize) &&
         (std::get<2>(data).size() <= kMaxMatrixSize &&
          std::get<2>(data).size() == std::get<0>(data) * std::get<1>(data)) &&
         (!std::get<2>(data).empty());
}

bool FatehovKMatrixMaxElemMPI::PreProcessingImpl() {
  return true;
}

bool FatehovKMatrixMaxElemMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t rows = 0;
  size_t columns = 0;
  if (world_rank == 0) {
    auto &data = GetInput();
    rows = std::get<0>(data);
    columns = std::get<1>(data);
  }

  MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  size_t total_elems = rows * columns;
  size_t elems_per_proc = total_elems / world_size;
  size_t remainder = total_elems % world_size;

  std::vector<int> send_counts(world_size);
  std::vector<int> displacements(world_size);

  for (int i = 0; i < world_size; ++i) {
    send_counts[i] = static_cast<int>(elems_per_proc) + (std::cmp_less(i, remainder) ? 1 : 0);
    displacements[i] = (i == 0) ? 0 : (displacements[i - 1] + send_counts[i - 1]);
  }

  std::vector<double> local_data(send_counts[world_rank]);
  const std::vector<double> *full_matrix_ptr = nullptr;

  if (world_rank == 0) {
    auto &data = GetInput();
    full_matrix_ptr = &std::get<2>(data);
  }

  MPI_Scatterv((world_rank == 0) ? (*full_matrix_ptr).data() : nullptr, send_counts.data(), displacements.data(),
               MPI_DOUBLE, local_data.data(), send_counts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_max = -std::numeric_limits<double>::max();
  for (const auto &value : local_data) {
    local_max = std::max(value, local_max);
  }

  double global_max = NAN;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool FatehovKMatrixMaxElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_max_elem
