#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"

namespace khruev_a_min_elem_vec {

KhruevAMinElemVecMPI::KhruevAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // mpi scoreboard
  GetInput() = in;                       // dannie doljna bit vidna vsem func rodytelya and stabilizaciya
  GetOutput() = 0;
}

bool KhruevAMinElemVecMPI::ValidationImpl() {  // input check
  return GetOutput() == 0;
}

bool KhruevAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool KhruevAMinElemVecMPI::RunImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    GetOutput() = INT_MAX;
    return true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = static_cast<int>(input.size());
  int int_part = n / size;
  int remainder = n % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = int_part + (i < remainder ? 1 : 0);
    displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
  }

  std::vector<int> local_chunk(sendcounts[rank] > 0 ? sendcounts[rank] : 1);
  MPI_Scatterv(input.data(), sendcounts.data(), displs.data(), MPI_INT,
               sendcounts[rank] > 0 ? local_chunk.data() : nullptr, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = INT_MAX;
  if (sendcounts[rank] > 0) {
    local_min = *std::min_element(local_chunk.begin(), local_chunk.begin() + sendcounts[rank]);
  }

  int global_min = 0;

  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_min, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_min;

  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
