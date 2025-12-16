#include "akimov_i_words_string_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <vector>

#include "akimov_i_words_string_count/common/include/common.hpp"

namespace akimov_i_words_string_count {

namespace {

inline bool IsSpaceChar(char ch) {
  return ch == ' ' || ch == '\n' || ch == '\t';
}

int CountWordsInBuffer(const InType &buf) {
  int count = 0;
  bool in_word = false;

  for (char c : buf) {
    if (IsSpaceChar(c)) {
      in_word = false;
    } else {
      if (!in_word) {
        count++;
        in_word = true;
      }
    }
  }
  return count;
}

}  // namespace

AkimovIWordsStringCountMPI::AkimovIWordsStringCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIWordsStringCountMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (!GetInput().empty()) && (GetOutput() == 0);
  }
  return true;
}

bool AkimovIWordsStringCountMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  local_space_count_ = 0;
  global_space_count_ = 0;
  word_count_ = 0;
  local_buffer_.clear();
  input_buffer_.clear();

  std::size_t total = 0;
  if (rank == 0) {
    input_buffer_ = GetInput();
    total = input_buffer_.size();
  }

  int base = 0;
  int remainder = 0;
  if (rank == 0) {
    base = static_cast<int>(total / static_cast<std::size_t>(size));
    remainder = static_cast<int>(total % static_cast<std::size_t>(size));
  }

  MPI_Bcast(&base, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&remainder, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int my_count = base + ((rank < remainder) ? 1 : 0);
  local_buffer_.resize(my_count);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    counts[i] = base + ((i < remainder) ? 1 : 0);
  }
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  const char *sendbuf = nullptr;
  if (rank == 0 && !input_buffer_.empty()) {
    sendbuf = input_buffer_.data();
  }

  MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_CHAR, local_buffer_.data(), my_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  return true;
}

bool AkimovIWordsStringCountMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_word_count = CountWordsInBuffer(local_buffer_);

  char recv_prev_last = ' ';
  char send_last = local_buffer_.empty() ? ' ' : local_buffer_.back();

  int dest = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
  int src = (rank - 1 >= 0) ? rank - 1 : MPI_PROC_NULL;

  MPI_Sendrecv(&send_last, 1, MPI_CHAR, dest, 0, &recv_prev_last, 1, MPI_CHAR, src, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  if (!local_buffer_.empty()) {
    char first_char = local_buffer_.front();
    if (!IsSpaceChar(recv_prev_last) && !IsSpaceChar(first_char)) {
      local_word_count = std::max(local_word_count - 1, 0);
    }
  }

  MPI_Reduce(&local_word_count, &word_count_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&word_count_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = word_count_;

  return true;
}

bool AkimovIWordsStringCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_words_string_count
