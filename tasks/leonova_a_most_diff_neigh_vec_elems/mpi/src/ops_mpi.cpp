#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

LeonovaAMostDiffNeighVecElemsMPI::LeonovaAMostDiffNeighVecElemsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::tuple<int, int>(0, 0);
}

bool LeonovaAMostDiffNeighVecElemsMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = false;
  if (rank == 0) {
    is_valid = !GetInput().empty();
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool LeonovaAMostDiffNeighVecElemsMPI::PreProcessingImpl() {
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::RunImpl() {
  const auto &input_vec = GetInput();

  if (!ValidationImpl()) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = static_cast<int>(input_vec.size());
  ProcessWithMultipleProcesses(rank, size, total_size, input_vec);
  return true;
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessWithMultipleProcesses(int rank, int size, int total_size,
                                                                    const std::vector<int> &input_vec) {
  int actual_processes = std::min(size, total_size);
  int local_max_diff = -1;
  int local_first = 0;
  int local_second = 0;

  if (rank < actual_processes) {
    ProcessLocalData(rank, actual_processes, total_size, input_vec, local_max_diff, local_first, local_second, size);
  }

  GatherAndProcessResults(rank, actual_processes, local_max_diff, local_first, local_second, size);
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessLocalData(int rank, int actual_processes, int total_size,
                                                        const std::vector<int> &input_vec, int &local_max_diff,
                                                        int &local_first, int &local_second, int size) {
  int chunk_size = total_size / actual_processes;
  int remainder = total_size % actual_processes;

  std::vector<int> sizes(actual_processes);
  std::vector<int> offsets(actual_processes);

  int offset = 0;
  for (int i = 0; i < actual_processes; ++i) {
    sizes[i] = chunk_size + (i < remainder ? 1 : 0) + 1;
    offsets[i] = offset;
    offset += sizes[i] - 1;
  }

  if (actual_processes > 0) {
    sizes[actual_processes - 1] = total_size - offsets[actual_processes - 1];
  }

  int my_size = (rank < actual_processes) ? sizes[rank] : 0;

  if (my_size > 0) {
    std::vector<int> local_data(my_size);
    ReceiveLocalData(rank, actual_processes, input_vec, sizes, offsets, local_data, size);
    FindLocalMaxDiff(local_data, local_max_diff, local_first, local_second);
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::ReceiveLocalData(int rank, int actual_processes,
                                                        const std::vector<int> &input_vec,
                                                        const std::vector<int> &sizes, const std::vector<int> &offsets,
                                                        std::vector<int> &local_data, int size) {
  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);
  std::vector<int> recv_counts(size, 0);

  for (int i = 0; i < actual_processes; ++i) {
    send_counts[i] = sizes[i];
    displacements[i] = offsets[i];
    recv_counts[i] = sizes[i];
  }

  if (rank == 0) {
    if (!local_data.empty()) {
      std::copy(input_vec.begin() + offsets[0], input_vec.begin() + offsets[0] + sizes[0], local_data.begin());
    }
  }

  int recv_count = static_cast<int>(local_data.size());
  MPI_Scatterv((rank == 0) ? input_vec.data() : nullptr, send_counts.data(), displacements.data(), MPI_INT,
               local_data.data(), recv_count, MPI_INT, 0, MPI_COMM_WORLD);
}

void LeonovaAMostDiffNeighVecElemsMPI::FindLocalMaxDiff(const std::vector<int> &local_data, int &local_max_diff,
                                                        int &local_first, int &local_second) {
  if (local_data.size() == 1) {
    local_max_diff = 0;
    local_first = local_data[0];
    local_second = local_data[0];
    return;
  }

  for (int index = 0; index < static_cast<int>(local_data.size()) - 1; ++index) {
    int64_t diff = static_cast<int64_t>(local_data[index]) - static_cast<int64_t>(local_data[index + 1]);
    int64_t curr_diff_ll = std::llabs(diff);  // safe abs

    int curr_diff = (curr_diff_ll <= static_cast<int64_t>(std::numeric_limits<int>::max()))
                        ? static_cast<int>(curr_diff_ll)
                        : std::numeric_limits<int>::max();

    if (curr_diff > local_max_diff) {
      local_max_diff = curr_diff;
      local_first = local_data[index];
      local_second = local_data[index + 1];
    }
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::GatherAndProcessResults(int rank, int actual_processes, int local_max_diff,
                                                               int local_first, int local_second, int size) {
  struct ProcessResult {
    int diff;
    int first;
    int second;
  };

  ProcessResult local_result{.diff = local_max_diff, .first = local_first, .second = local_second};
  std::vector<ProcessResult> all_results(size);

  MPI_Gather(&local_result, 3, MPI_INT, all_results.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int global_max_diff = -1;
    int best_index = -1;

    for (int index = 0; index < actual_processes; ++index) {
      if (all_results[index].diff > global_max_diff) {
        global_max_diff = all_results[index].diff;
        best_index = index;
      }
    }

    if (best_index != -1) {
      GetOutput() = std::make_tuple(all_results[best_index].first, all_results[best_index].second);
    } else {
      GetOutput() = std::make_tuple(0, 0);
    }
  }

  BroadcastResult(rank);
}

void LeonovaAMostDiffNeighVecElemsMPI::BroadcastResult(int rank) {
  std::array<int, 2> result_data{0, 0};

  if (rank == 0) {
    result_data[0] = std::get<0>(GetOutput());
    result_data[1] = std::get<1>(GetOutput());
  }

  MPI_Bcast(result_data.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = std::make_tuple(result_data[0], result_data[1]);
  }
}

bool LeonovaAMostDiffNeighVecElemsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_most_diff_neigh_vec_elems
