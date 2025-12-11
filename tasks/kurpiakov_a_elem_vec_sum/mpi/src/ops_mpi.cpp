#include "kurpiakov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <utility>
#include <vector>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"

namespace kurpiakov_a_elem_vec_sum {

KurpiakovAElemVecSumMPI::KurpiakovAElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KurpiakovAElemVecSumMPI::ValidationImpl() {
  bool res = (GetOutput() == 0) && (std::cmp_equal((std::get<1>(GetInput()).size()), std::get<0>(GetInput())));
  return res;
}

bool KurpiakovAElemVecSumMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KurpiakovAElemVecSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = std::get<0>(GetInput());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    GetOutput() = 0LL;
    return true;
  }

  std::vector<int> batch(size);
  std::vector<int> displs(size);
  int batch_size = total_size / size;
  int remainder = total_size % size;
  for (int i = 0; i < size; ++i) {
    batch[i] = batch_size + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + batch[i - 1];
  }

  int local_size = batch[rank];
  std::vector<int> local_data(local_size);

  int *sendbuf = nullptr;
  if (rank == 0) {
    sendbuf = const_cast<int *>(std::get<1>(GetInput()).data());
  }

  MPI_Scatterv(sendbuf, batch.data(), displs.data(), MPI_INT, local_data.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  OutType local_sum = 0LL;
  for (int i = 0; i < local_size; ++i) {
    local_sum += static_cast<OutType>(local_data[i]);
  }

  OutType global_sum = 0LL;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_sum, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool KurpiakovAElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_elem_vec_sum
