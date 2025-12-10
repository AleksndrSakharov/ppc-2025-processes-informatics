#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "gasenin_l_lex_dif/common/include/common.hpp"

namespace {

struct LocalDiff {
  size_t diff_pos;
  int result;
};

LocalDiff FindLocalDifference(const std::vector<char> &s1_chunk, const std::vector<char> &s2_chunk,
                              size_t global_start) {
  for (size_t i = 0; i < s1_chunk.size(); ++i) {
    if (s1_chunk[i] != s2_chunk[i]) {
      return {.diff_pos = global_start + i, .result = (s1_chunk[i] < s2_chunk[i]) ? -1 : 1};
    }
  }
  return {.diff_pos = std::string::npos, .result = 0};
}

void CalculateDistribution(int min_len, int size, std::vector<int> &sendcounts, std::vector<int> &displs) {
  int remainder = min_len % size;
  int chunk_size = min_len / size;
  int prefix_sum = 0;

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
    displs[i] = prefix_sum;
    prefix_sum += sendcounts[i];
  }
}

int GetFinalResultByLength(int len1, int len2) {
  if (len1 < len2) {
    return -1;
  }
  if (len1 > len2) {
    return 1;
  }
  return 0;
}

}  // namespace

namespace gasenin_l_lex_dif {

GaseninLLexDifMPI::GaseninLLexDifMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLLexDifMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &[str1, str2] = GetInput();
    return str1.length() <= 100000000 && str2.length() <= 100000000;
  }
  return true;
}

bool GaseninLLexDifMPI::PreProcessingImpl() {
  return true;
}

bool GaseninLLexDifMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input_data = GetInput();
  const std::string &str1 = input_data.first;
  const std::string &str2 = input_data.second;

  std::vector<int> lengths(2, 0);
  if (rank == 0) {
    lengths[0] = static_cast<int>(str1.length());
    lengths[1] = static_cast<int>(str2.length());
  }

  MPI_Bcast(lengths.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  int len1 = lengths[0];
  int len2 = lengths[1];
  int min_len = std::min(len1, len2);

  if (min_len == 0 && len1 == len2) {
    GetOutput() = 0;
    return true;
  }

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  CalculateDistribution(min_len, size, sendcounts, displs);

  int local_count = sendcounts[rank];
  auto global_start = static_cast<size_t>(displs[rank]);

  std::vector<char> local_str1(local_count);
  std::vector<char> local_str2(local_count);

  const char *sendbuf1 = (rank == 0) ? str1.data() : nullptr;
  const char *sendbuf2 = (rank == 0) ? str2.data() : nullptr;

  MPI_Scatterv(sendbuf1, sendcounts.data(), displs.data(), MPI_CHAR, local_str1.data(), local_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(sendbuf2, sendcounts.data(), displs.data(), MPI_CHAR, local_str2.data(), local_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  LocalDiff local_diff = FindLocalDifference(local_str1, local_str2, global_start);

  int local_diff_pos_int = (local_diff.diff_pos == std::string::npos) ? min_len : static_cast<int>(local_diff.diff_pos);
  int global_min_pos_int = min_len;

  MPI_Allreduce(&local_diff_pos_int, &global_min_pos_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int result_for_sum = 0;
  if (local_diff_pos_int == global_min_pos_int && local_diff_pos_int < min_len) {
    result_for_sum = local_diff.result;
  }

  int final_result = 0;
  MPI_Allreduce(&result_for_sum, &final_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (global_min_pos_int == min_len) {
    final_result = GetFinalResultByLength(len1, len2);
  }

  GetOutput() = final_result;
  return true;
}

bool GaseninLLexDifMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_lex_dif
