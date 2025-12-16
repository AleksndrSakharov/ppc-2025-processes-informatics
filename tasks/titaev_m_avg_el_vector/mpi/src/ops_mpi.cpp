#include "titaev_m_avg_el_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "titaev_m_avg_el_vector/common/include/common.hpp"

namespace titaev_m_avg_el_vector {

TitaevMElemVecsAvgMPI::TitaevMElemVecsAvgMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TitaevMElemVecsAvgMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevMElemVecsAvgMPI::PreProcessingImpl() {
  return true;
}

bool TitaevMElemVecsAvgMPI::RunImpl() {
  const auto &vec = GetInput();

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  size_t n = vec.size();
  int base = static_cast<int>(n / size);
  int rem = static_cast<int>(n % size);

  for (int i = 0; i < size; ++i) {
    counts[i] = base;
    if (i < rem) {
      counts[i]++;
    }
  }

  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<int> local(counts[rank]);

  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT, local.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  double local_sum = 0.0;
  for (int x : local) {
    local_sum += x;
  }

  double global_sum = 0.0;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum / static_cast<double>(n);
  return true;
}

bool TitaevMElemVecsAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_avg_el_vector
