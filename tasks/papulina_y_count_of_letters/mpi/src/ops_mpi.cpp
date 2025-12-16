#include "papulina_y_count_of_letters/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "papulina_y_count_of_letters/common/include/common.hpp"

namespace papulina_y_count_of_letters {

PapulinaYCountOfLettersMPI::PapulinaYCountOfLettersMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procNum_);
}
int PapulinaYCountOfLettersMPI::CountOfLetters(const char *s, const int &n) {
  int k = 0;
  if (n <= 0) {
    return 0;
  }
  for (int i = 0; i < n; i++) {
    char c = s[i];
    if (isalpha(c) != 0) {
      k++;
    }
  }
  return k;
}
bool PapulinaYCountOfLettersMPI::ValidationImpl() {
  return procNum_ > 0;
}

bool PapulinaYCountOfLettersMPI::PreProcessingImpl() {
  return true;
}

bool PapulinaYCountOfLettersMPI::RunImpl() {
  int proc_rank = 0;
  int result = 0;
  std::string part_of_string;  // части строки, которая будет обрабатываться потоком
  unsigned int len = 0;        // предполагаемая длина обрабатываемой части
  unsigned int remainder = 0;  // остаток, если длина строки не кратна числу потоков
  unsigned int true_len = 0;   // реальная длина обрабатываемой части
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    std::string s = GetInput();

    len = s.size() / procNum_;
    remainder = s.size() % procNum_;
    MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&remainder, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int begin_0 = (0 * len) + std::min(static_cast<unsigned int>(0), remainder);
    unsigned int end_0 = ((0 + 1) * len) + std::min(static_cast<unsigned int>(0 + 1), remainder);
    true_len = end_0 - begin_0;
    part_of_string = (true_len > 0) ? s.substr(begin_0, true_len) : "";

    for (int i = 1; i < procNum_; i++) {
      unsigned int begin = (i * len) + std::min(static_cast<unsigned int>(i), remainder);
      unsigned int end = ((i + 1) * len) + std::min(static_cast<unsigned int>(i + 1), remainder);
      unsigned int pre_true_len = end - begin;  // предварительная длина обрабатываемой части

      MPI_Send(&pre_true_len, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
      if (end - begin != 0) {
        MPI_Send(s.substr(begin, pre_true_len).data(), static_cast<int>(pre_true_len), MPI_CHAR, i, 1, MPI_COMM_WORLD);
      } else {
        MPI_Send("", 0, MPI_CHAR, i, 1, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&remainder, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Recv(&true_len, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (true_len > 0) {
      part_of_string.resize(true_len);
    }
    MPI_Recv(part_of_string.data(), static_cast<int>(true_len), MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int local_result = CountOfLetters(part_of_string.data(), static_cast<int>(part_of_string.size()));
  MPI_Reduce(&local_result, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PapulinaYCountOfLettersMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace papulina_y_count_of_letters
