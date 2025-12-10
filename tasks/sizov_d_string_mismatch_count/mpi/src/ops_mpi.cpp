#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountMPI::SizovDStringMismatchCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountMPI::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return !a.empty() && (a.size() == b.size());
}

bool SizovDStringMismatchCountMPI::PreProcessingImpl() {
  const auto &[a, b] = GetInput();
  str_a_ = a;
  str_b_ = b;
  return true;
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  int rank = 0;
  int world_size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(str_a_.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base = total_size / world_size;
  const int remainder = total_size % world_size;

  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);

  int offset = 0;
  for (int i = 0; i < world_size; i++) {
    counts[i] = base + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  const int local_size = counts[rank];
  std::string local_a(local_size, '\0');
  std::string local_b(local_size, '\0');

  MPI_Scatterv(str_a_.data(), counts.data(), displs.data(), MPI_CHAR, local_a.data(), local_size, MPI_CHAR, 0,
               MPI_COMM_WORLD);
  MPI_Scatterv(str_b_.data(), counts.data(), displs.data(), MPI_CHAR, local_b.data(), local_size, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int local_result = 0;
  for (int i = 0; i < local_size; ++i) {
    if (local_a[i] != local_b[i]) {
      ++local_result;
    }
  }

  int global_result = 0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_string_mismatch_count
