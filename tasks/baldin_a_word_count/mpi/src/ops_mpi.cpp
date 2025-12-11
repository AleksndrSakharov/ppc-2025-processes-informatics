#include "baldin_a_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "baldin_a_word_count/common/include/common.hpp"

namespace baldin_a_word_count {

BaldinAWordCountMPI::BaldinAWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaldinAWordCountMPI::ValidationImpl() {
  return true;
}

bool BaldinAWordCountMPI::PreProcessingImpl() {
  return true;
}

namespace {

bool IsWordChar(char c) {
  return ((std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '-' || c == '_');
}

size_t CountWords(const std::string &text) {
  size_t count = 0;
  bool in_word = false;
  for (char c : text) {
    if (IsWordChar(c)) {
      if (!in_word) {
        in_word = true;
        count++;
      }
    } else {
      in_word = false;
    }
  }
  return count;
}

size_t CountLocalWords(const std::vector<char> &local_buf, int part) {
  size_t local_cnt = 0;
  bool in_word = false;
  for (int i = 0; i < part; i++) {
    if (IsWordChar(local_buf[i])) {
      if (!in_word) {
        in_word = true;
        local_cnt++;
      }
    } else {
      in_word = false;
    }
  }

  if (in_word && (IsWordChar(local_buf[part]))) {
    local_cnt--;
  }

  return local_cnt;
}

}  // namespace

bool BaldinAWordCountMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::string &input = GetInput();

  if (input.empty()) {
    GetOutput() = 0;
    return true;
  }

  if (static_cast<size_t>(world_size) > input.size()) {
    if (rank == 0) {
      GetOutput() = CountWords(input);
    }
    MPI_Bcast(static_cast<void *>(&GetOutput()), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    return true;
  }

  size_t rem = input.size() % static_cast<size_t>(world_size);
  input.append(static_cast<size_t>(world_size) - rem + 1, ' ');
  size_t part = input.size() / static_cast<size_t>(world_size);

  std::vector<int> send_counts(world_size);
  std::vector<int> displs(world_size);

  if (rank == 0) {
    for (int i = 0; i < world_size; i++) {
      displs[i] = static_cast<int>(i * part);
      send_counts[i] = static_cast<int>(part + 1);
    }
  }

  std::vector<char> local_buf(part + 1);

  MPI_Scatterv(input.data(), send_counts.data(), displs.data(), MPI_CHAR, local_buf.data(), static_cast<int>(part + 1),
               MPI_CHAR, 0, MPI_COMM_WORLD);

  size_t local_cnt = CountLocalWords(local_buf, static_cast<int>(part));

  size_t global_cnt = 0;
  MPI_Allreduce(&local_cnt, &global_cnt, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_cnt;

  return true;
}

bool BaldinAWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
