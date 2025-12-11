#include "shkrebko_m_count_char_freq/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>
#include <vector>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"

namespace shkrebko_m_count_char_freq {

ShkrebkoMCountCharFreqMPI::ShkrebkoMCountCharFreqMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrebkoMCountCharFreqMPI::ValidationImpl() {
  return (!std::get<0>(GetInput()).empty()) && (std::get<1>(GetInput()).length() == 1);
}

bool ShkrebkoMCountCharFreqMPI::PreProcessingImpl() {
  return true;
}

bool ShkrebkoMCountCharFreqMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string input_text;
  char target_char = 0;
  int total_size = 0;

  if (rank == 0) {
    input_text = std::get<0>(GetInput());
    target_char = std::get<1>(GetInput())[0];
    total_size = static_cast<int>(input_text.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&target_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  const int base = total_size / size;
  const int remainder = total_size % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    sendcounts[i] = base + (i < remainder ? 1 : 0);
    displs[i] = (i * base) + std::min(i, remainder);
  }

  int local_size = sendcounts[rank];
  std::vector<char> local_data(local_size);

  MPI_Scatterv(rank == 0 ? input_text.data() : nullptr, sendcounts.data(), displs.data(), MPI_CHAR, local_data.data(),
               sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_count = 0;
  for (char c : local_data) {
    if (c == target_char) {
      local_count++;
    }
  }

  int global_result = 0;
  MPI_Allreduce(&local_count, &global_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}

bool ShkrebkoMCountCharFreqMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkrebko_m_count_char_freq
