#include "kutergin_a_closest_pair/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"

namespace kutergin_a_closest_pair {

KuterginAClosestPairMPI::KuterginAClosestPairMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -1;
}

bool KuterginAClosestPairMPI::ValidationImpl() {
  return true;
}

bool KuterginAClosestPairMPI::PreProcessingImpl() {
  return true;
}

namespace {

std::vector<int> DistributeData(int rank, int size, int n, const std::vector<int> &v) {
  int local_size = n / size;
  int remainder = n % size;

  int start = (rank * local_size) + std::min(rank, remainder);
  int end = start + local_size + (rank < remainder ? 1 : 0);

  if (rank == size - 1) {
    end = n;
  }

  std::vector<int> local_data(end - start);
  if (rank == 0) {
    std::copy(v.begin() + start, v.begin() + end, local_data.begin());

    for (int i = 1; i < size; ++i) {
      int other_start = (i * local_size) + std::min(i, remainder);
      int other_end = other_start + local_size + (i < remainder ? 1 : 0);
      if (i == size - 1) {
        other_end = n;
      }

      MPI_Send(v.data() + other_start, other_end - other_start, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_data.data(), static_cast<int>(local_data.size()), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return local_data;
}

int FindLocalMin(const std::vector<int> &local_data, int start_idx, int &found_idx) {
  int local_min = std::numeric_limits<int>::max();
  found_idx = -1;

  for (int i = 0; i < static_cast<int>(local_data.size()) - 1; ++i) {
    int diff = std::abs(local_data[i + 1] - local_data[i]);
    if (diff < local_min) {
      local_min = diff;
      found_idx = start_idx + i;
    }
  }

  return local_min;
}

int CalculateStartIndex(int rank, int size, int n) {
  int local_size = n / size;
  int remainder = n % size;
  return (rank * local_size) + std::min(rank, remainder);
}

int CalculateEndIndex(int rank, int size, int n) {
  int local_size = n / size;
  int remainder = n % size;
  int end = CalculateStartIndex(rank, size, n) + local_size + (rank < remainder ? 1 : 0);
  if (rank == size - 1) {
    end = n;
  }
  return end;
}

int CheckBoundary(int rank, int size, int end, int n, const std::vector<int> &v, const std::vector<int> &local_data,
                  int current_min, int &current_idx) {
  if (rank < size - 1 && end < n) {
    int boundary_diff = std::abs(v[end] - local_data.back());
    if (boundary_diff < current_min) {
      current_min = boundary_diff;
      current_idx = end - 1;
    }
  }
  return current_min;
}

}  // namespace

bool KuterginAClosestPairMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &v = GetInput();
  int n = static_cast<int>(v.size());

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 2) {
    GetOutput() = -1;
    return true;
  }

  auto local_data = DistributeData(rank, size, n, v);
  if (local_data.empty()) {
    GetOutput() = -1;
    return true;
  }

  int start_idx = CalculateStartIndex(rank, size, n);
  int local_idx = -1;
  int local_min = FindLocalMin(local_data, start_idx, local_idx);

  int end = CalculateEndIndex(rank, size, n);
  local_min = CheckBoundary(rank, size, end, n, v, local_data, local_min, local_idx);

  struct MinIndex {
    int val = 0;
    int idx = -1;
  };

  MinIndex local_result;
  local_result.val = local_min;
  local_result.idx = local_idx;

  MinIndex global_result;

  MPI_Allreduce(&local_result, &global_result, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

  GetOutput() = global_result.idx;
  return true;
}

bool KuterginAClosestPairMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_closest_pair
