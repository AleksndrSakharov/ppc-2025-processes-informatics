#include "kruglova_a_max_diff_adjacent/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"

namespace kruglova_a_max_diff_adjacent {

KruglovaAMaxDiffAdjacentMPI::KruglovaAMaxDiffAdjacentMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0F;
}

bool KruglovaAMaxDiffAdjacentMPI::ValidationImpl() {
  return true;
}

bool KruglovaAMaxDiffAdjacentMPI::PreProcessingImpl() {
  return true;
}

float KruglovaAMaxDiffAdjacentMPI::LocalMaxDiff(const std::vector<float> &local_vec) {
  float local_max = 0.0F;

  if (local_vec.size() >= 2) {
    for (size_t i = 1; i < local_vec.size(); ++i) {
      float diff = std::abs(local_vec[i] - local_vec[i - 1]);
      local_max = std::max(diff, local_max);
    }
  }
  return local_max;
}

bool KruglovaAMaxDiffAdjacentMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &vec = GetInput();
  int n = 0;

  if (rank == 0) {
    n = static_cast<int>(vec.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 2) {
    if (rank == 0) {
      GetOutput() = 0.0F;
    }
    return true;
  }

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int base = n / size;
  int rem = n % size;

  for (int proc = 0; proc < size; ++proc) {
    int count = base + (proc < rem ? 1 : 0);

    displs[proc] = (proc * base) + std::min(proc, rem);

    if (displs[proc] + count < n) {
      sendcounts[proc] = count + 1;
    } else {
      sendcounts[proc] = count;
    }
  }

  int local_count = sendcounts[rank];
  std::vector<float> local_vec(local_count);

  MPI_Scatterv(rank == 0 ? vec.data() : nullptr, sendcounts.data(), displs.data(), MPI_FLOAT, local_vec.data(),
               local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

  float local_max = 0.0F;
  local_max = LocalMaxDiff(local_vec);

  float global_max = 0.0F;

  MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_max, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool KruglovaAMaxDiffAdjacentMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_max_diff_adjacent
