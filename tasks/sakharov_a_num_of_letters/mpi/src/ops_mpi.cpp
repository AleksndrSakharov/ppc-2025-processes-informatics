#include "sakharov_a_num_of_letters/mpi/include/ops_mpi.hpp"

#include <mpi.h>

// #include <numeric>
// #include <vector>
#include <algorithm>
#include <cctype>
#include <utility>
#include <vector>

#include "sakharov_a_num_of_letters/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sakharov_a_num_of_letters {

SakharovANumberOfLettersMPI::SakharovANumberOfLettersMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SakharovANumberOfLettersMPI::ValidationImpl() {
  return std::cmp_equal(std::get<1>(GetInput()).size(), std::get<0>(GetInput()));
}

bool SakharovANumberOfLettersMPI::PreProcessingImpl() {
  return true;
}

bool SakharovANumberOfLettersMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size <= 0) {
    return false;
  }

  int size_of_string = 0;
  char *send_buffer = nullptr;

  if (world_rank == 0) {
    size_of_string = std::get<0>(GetInput());
    const std::string &string_of_letters = std::get<1>(GetInput());
    if (size_of_string > 0) {
      send_buffer = const_cast<char *>(string_of_letters.data());
    }
  }

  MPI_Bcast(&size_of_string, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (size_of_string < 0) {
    return false;
  }

  int base_chunk = size_of_string / world_size;
  int remainder = size_of_string % world_size;

  std::vector<int> send_counts(world_size);
  std::vector<int> displs(world_size);
  for (int r = 0; r < world_size; r++) {
    send_counts[r] = base_chunk + (r < remainder ? 1 : 0);
    displs[r] = r * base_chunk + std::min(r, remainder);
  }

  int local_count = send_counts[world_rank];
  std::string local_string;
  local_string.resize(local_count);

  char *local_buffer = local_count > 0 ? local_string.data() : nullptr;

  MPI_Scatterv(send_buffer, send_counts.data(), displs.data(), MPI_CHAR, local_buffer, local_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int local_letters = 0;
  for (int i = 0; i < local_count; i++) {
    char c = local_buffer[i];
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
      local_letters++;
    }
  }

  int global_letters = 0;
  MPI_Allreduce(&local_letters, &global_letters, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_letters;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SakharovANumberOfLettersMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_num_of_letters
