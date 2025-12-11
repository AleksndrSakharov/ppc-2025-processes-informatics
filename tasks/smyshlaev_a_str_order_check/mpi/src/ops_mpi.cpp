#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>
#include <vector>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"

namespace smyshlaev_a_str_order_check {

namespace {

int CompareBuffers(const char *s1, const char *s2, int len) {
  for (int i = 0; i < len; ++i) {
    if (s1[i] < s2[i]) {
      return -1;
    }
    if (s1[i] > s2[i]) {
      return 1;
    }
  }
  return 0;
}

int ResolveResult(int diff_res, int len1, int len2) {
  if (diff_res != 0) {
    return diff_res;
  }
  if (len1 < len2) {
    return -1;
  }
  if (len1 > len2) {
    return 1;
  }
  return 0;
}

void CalculateDistribution(int total_len, int proc_count, std::vector<int> &counts, std::vector<int> &offsets) {
  const int chunk = total_len / proc_count;
  const int remainder = total_len % proc_count;
  int offset = 0;
  for (int i = 0; i < proc_count; i++) {
    counts[i] = chunk + (i < remainder ? 1 : 0);
    offsets[i] = offset;
    offset += counts[i];
  }
}

}  // namespace

SmyshlaevAStrOrderCheckMPI::SmyshlaevAStrOrderCheckMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }

  GetOutput() = 0;
}

bool SmyshlaevAStrOrderCheckMPI::ValidationImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::RunSequential(int min_len, int len1, int len2) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int res = 0;
  if (rank == 0) {
    const auto &str1 = GetInput().first;
    const auto &str2 = GetInput().second;
    int cmp = CompareBuffers(str1.data(), str2.data(), min_len);
    res = ResolveResult(cmp, len1, len2);
  }

  MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = res;
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int len1 = 0;
  int len2 = 0;

  if (rank == 0) {
    const auto &input_data = GetInput();
    len1 = static_cast<int>(input_data.first.length());
    len2 = static_cast<int>(input_data.second.length());
  }

  MPI_Bcast(&len1, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&len2, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int min_len = static_cast<int>(std::min(len1, len2));

  if (proc_count > min_len) {
    return RunSequential(min_len, len1, len2);
  }

  std::vector<int> sendcounts(proc_count);
  std::vector<int> offsets(proc_count);
  CalculateDistribution(min_len, proc_count, sendcounts, offsets);

  int local_size = sendcounts[rank];
  std::vector<char> local_str1(local_size);
  std::vector<char> local_str2(local_size);

  const char *s1_ptr = (rank == 0) ? GetInput().first.data() : nullptr;
  const char *s2_ptr = (rank == 0) ? GetInput().second.data() : nullptr;

  MPI_Scatterv(s1_ptr, sendcounts.data(), offsets.data(), MPI_CHAR, local_str1.data(), local_size, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(s2_ptr, sendcounts.data(), offsets.data(), MPI_CHAR, local_str2.data(), local_size, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int local_result = CompareBuffers(local_str1.data(), local_str2.data(), local_size);

  std::vector<int> all_results(proc_count);
  MPI_Allgather(&local_result, 1, MPI_INT, all_results.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int global_result = 0;
  for (int res : all_results) {
    if (res != 0) {
      global_result = res;
      break;
    }
  }

  GetOutput() = ResolveResult(global_result, len1, len2);

  return true;
}

bool SmyshlaevAStrOrderCheckMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_str_order_check
