#include "maslova_u_char_frequency_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "maslova_u_char_frequency_count/common/include/common.hpp"

namespace maslova_u_char_frequency_count {

MaslovaUCharFrequencyCountMPI::MaslovaUCharFrequencyCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MaslovaUCharFrequencyCountMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int flag = 0;  // 0 - всё ок, 1 - ошибка
  if (rank == 0) {
    if (GetInput().first.size() > static_cast<size_t>(INT_MAX)) {
      flag = 1;
    }
  }
  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (flag == 0);
}

bool MaslovaUCharFrequencyCountMPI::PreProcessingImpl() {
  return true;
}

bool MaslovaUCharFrequencyCountMPI::RunImpl() {
  int rank = 0;
  int proc_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // id процесса
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);  // количество процессов

  std::string input_string;
  char input_char = 0;
  size_t input_str_size = 0;

  if (rank == 0) {
    input_string = GetInput().first;
    input_char = GetInput().second;
    input_str_size = input_string.size();  // получили данные
  }

  uint64_t size_for_mpi = 0;
  if (rank == 0) {
    size_for_mpi = static_cast<uint64_t>(input_str_size);  // явное приведение перед передачей
  }

  MPI_Bcast(&size_for_mpi, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);  // отправляем размер строки

  if (rank != 0) {
    input_str_size = static_cast<size_t>(size_for_mpi);  // возращаем обратно для удобного использования в дальнейшем
  }

  if (input_str_size == 0) {
    GetOutput() = 0;  // ставим для всех процессов
    return true;
  }

  MPI_Bcast(&input_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);  // отправляем нужный символ

  std::vector<int> send_counts(proc_size);  // здесь размеры всех порций
  std::vector<int> displs(proc_size);       // смещения
  if (rank == 0) {
    size_t part = input_str_size / proc_size;
    size_t rem = input_str_size % proc_size;
    for (size_t i = 0; std::cmp_less(i, proc_size); ++i) {
      send_counts[i] = static_cast<int>(part + (i < rem ? 1 : 0));  // общий размер, включающий остаток, если он входит
    }
    displs[0] = 0;
    for (size_t i = 1; std::cmp_less(i, proc_size); ++i) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size, MPI_INT, 0, MPI_COMM_WORLD);  // отправляем размеры порций
  std::vector<char> local_str(send_counts[rank]);
  MPI_Scatterv((rank == 0) ? input_string.data() : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_str.data(), static_cast<int>(local_str.size()), MPI_CHAR, 0, MPI_COMM_WORLD  // распределяем данные
  );

  size_t local_count = std::count(local_str.begin(), local_str.end(), input_char);
  auto local_count_for_mpi = static_cast<uint64_t>(local_count);
  uint64_t global_count = 0;
  MPI_Allreduce(&local_count_for_mpi, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);  // собрали данные со всех процессов

  GetOutput() = static_cast<size_t>(global_count);  // вывели результат, при этом приведя его к нужному нам типу

  return true;
}

bool MaslovaUCharFrequencyCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_char_frequency_count
