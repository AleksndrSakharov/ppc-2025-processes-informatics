#include "../include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "../../common/include/common.hpp"

namespace shekhirev_v_char_freq_mpi {

CharFreqMPI::CharFreqMPI(const shekhirev_v_char_freq_seq::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool CharFreqMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetInput().str.size() <= static_cast<size_t>(INT_MAX);
  }
  return true;
}

bool CharFreqMPI::PreProcessingImpl() {
  return true;
}

bool CharFreqMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_len = 0;
  char target = 0;

  if (rank == 0) {
    total_len = static_cast<int>(GetInput().str.size());
    target = GetInput().target;
  }

  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&target, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (total_len == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  const int delta = total_len / size;
  const int remainder = total_len % size;
  int my_count = delta + (rank < remainder ? 1 : 0);

  std::vector<char> local_data(my_count);

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    int current_displ = 0;
    for (int i = 0; i < size; ++i) {
      send_counts[i] = delta + (i < remainder ? 1 : 0);
      displs[i] = current_displ;
      current_displ += send_counts[i];
    }
  }

  const char *sendbuf = (rank == 0) ? GetInput().str.data() : nullptr;

  MPI_Scatterv(sendbuf, send_counts.data(), displs.data(), MPI_CHAR, local_data.data(), my_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int local_res = static_cast<int>(std::count(local_data.begin(), local_data.end(), target));

  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &local_res, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    GetOutput() = local_res;
  } else {
    MPI_Reduce(&local_res, nullptr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool CharFreqMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_char_freq_mpi
