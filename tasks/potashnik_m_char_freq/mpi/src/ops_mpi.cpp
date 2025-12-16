#include "potashnik_m_char_freq/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <string>
#include <vector>

#include "potashnik_m_char_freq/common/include/common.hpp"

namespace potashnik_m_char_freq {

PotashnikMCharFreqMPI::PotashnikMCharFreqMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }

  GetOutput() = 0;
}

bool PotashnikMCharFreqMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  return !std::get<0>(GetInput()).empty();
}

bool PotashnikMCharFreqMPI::PreProcessingImpl() {
  return true;
}

bool PotashnikMCharFreqMPI::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string str;
  char chr = 0;
  int string_size = 0;

  if (rank == 0) {
    str = std::get<0>(GetInput());
    chr = std::get<1>(GetInput());
    string_size = static_cast<int>(str.size());
  }

  MPI_Bcast(&string_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&chr, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  int block_size = string_size / world_size;
  int remainder = string_size % world_size;

  std::vector<int> local_sizes(world_size, 0);
  std::vector<int> local_start_positions(world_size, 0);

  if (rank == 0) {
    for (int i = 0; i < world_size; i++) {
      if (remainder > i) {
        local_sizes[i] = block_size + 1;
      } else {
        local_sizes[i] = block_size;
      }
    }

    for (int i = 1; i < world_size; i++) {
      local_start_positions[i] = local_start_positions[i - 1] + local_sizes[i - 1];
    }
  }

  int cur_count = 0;
  MPI_Scatter(local_sizes.data(), 1, MPI_INT, &cur_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::string cur_str(cur_count, '\0');
  if (rank == 0) {
    std::vector<char> temp_str(str.begin(), str.end());  // Dont work without const
    MPI_Scatterv(temp_str.data(), local_sizes.data(), local_start_positions.data(), MPI_CHAR, cur_str.data(), cur_count,
                 MPI_CHAR, 0, MPI_COMM_WORLD);
    ;
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_CHAR, cur_str.data(), cur_count, MPI_CHAR, 0,
                 MPI_COMM_WORLD);  // Just recieving data
  }

  int cur_res = 0;
  for (char c : cur_str) {
    if (c == chr) {
      cur_res++;
    }
  }

  int total_res = 0;
  MPI_Allreduce(&cur_res, &total_res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_res;

  return true;
}

bool PotashnikMCharFreqMPI::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_char_freq
