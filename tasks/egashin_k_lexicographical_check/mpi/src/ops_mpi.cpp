#include "egashin_k_lexicographical_check/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "egashin_k_lexicographical_check/common/include/common.hpp"

namespace egashin_k_lexicographical_check {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = false;
}

bool TestTaskMPI::ValidationImpl() {
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

void TestTaskMPI::CalculateDistribution(int size, int min_len, std::vector<int> &counts, std::vector<int> &displs) {
  int delta = min_len / size;
  int remainder = min_len % size;
  for (int i = 0; i < size; ++i) {
    counts[i] = delta + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }
}

int TestTaskMPI::CompareLocal(const std::vector<char> &s1, const std::vector<char> &s2, int count) {
  for (int i = 0; i < count; ++i) {
    auto c1 = static_cast<unsigned char>(s1[i]);
    auto c2 = static_cast<unsigned char>(s2[i]);
    if (c1 < c2) {
      return -1;
    }
    if (c1 > c2) {
      return 1;
    }
  }
  return 0;
}

bool TestTaskMPI::GetFinalDecision(const std::vector<int> &global_results, int s1_len, int s2_len) {
  for (int res : global_results) {
    if (res != 0) {
      return res == -1;
    }
  }
  return s1_len < s2_len;
}

bool TestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int min_len = 0;
  int s1_len = 0;
  int s2_len = 0;

  if (rank == 0) {
    s1_len = static_cast<int>(GetInput().first.size());
    s2_len = static_cast<int>(GetInput().second.size());
    min_len = std::min(s1_len, s2_len);
  }

  MPI_Bcast(&min_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (min_len == 0) {
    if (rank == 0) {
      GetOutput() = (s1_len < s2_len);
    }
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    CalculateDistribution(size, min_len, counts, displs);
  }

  int local_count = 0;
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> local_s1(local_count);
  std::vector<char> local_s2(local_count);
  const char *send_s1 = (rank == 0) ? GetInput().first.data() : nullptr;
  const char *send_s2 = (rank == 0) ? GetInput().second.data() : nullptr;

  if (local_count > 0 || rank == 0) {
    MPI_Scatterv(send_s1, counts.data(), displs.data(), MPI_CHAR, local_s1.data(), local_count, MPI_CHAR, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(send_s2, counts.data(), displs.data(), MPI_CHAR, local_s2.data(), local_count, MPI_CHAR, 0,
                 MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(send_s1, counts.data(), displs.data(), MPI_CHAR, local_s1.data(), 0, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(send_s2, counts.data(), displs.data(), MPI_CHAR, local_s2.data(), 0, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  int local_res = CompareLocal(local_s1, local_s2, local_count);

  std::vector<int> global_results(size);
  MPI_Gather(&local_res, 1, MPI_INT, global_results.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = GetFinalDecision(global_results, s1_len, s2_len);
  }
  return true;
}

}  // namespace egashin_k_lexicographical_check
