#include "votincev_d_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

VotincevDMatrixMultMPI::VotincevDMatrixMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// проверка входных данных
bool VotincevDMatrixMultMPI::ValidationImpl() {
  const auto &in = GetInput();
  int param_m = std::get<0>(in);
  int param_n = std::get<1>(in);
  int param_k = std::get<2>(in);
  const auto &matrix_a = std::get<3>(in);
  const auto &matrix_b = std::get<4>(in);

  return (param_m > 0 && param_n > 0 && param_k > 0 && static_cast<int>(matrix_a.size()) == (param_m * param_k) &&
          static_cast<int>(matrix_b.size()) == (param_k * param_n));
}

// препроцессинг
bool VotincevDMatrixMultMPI::PreProcessingImpl() {
  return true;
}

// главный метод MPI
bool VotincevDMatrixMultMPI::RunImpl() {
  // получаю кол-во процессов
  int process_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_n);

  // получаю ранг текущего
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // размерности матриц получают все процессы
  int m = 0;
  int n = 0;
  int k = 0;
  const auto &in = GetInput();
  m = std::get<0>(in);
  n = std::get<1>(in);
  k = std::get<2>(in);

  // если процессов больше чем строк в матрице A -
  // то активных процессов будет меньше (m)
  //  (потому что разедление по строкам)
  process_n = std::min(process_n, m);

  // "лишние" процессы не работают
  if (proc_rank >= process_n) {
    return true;
  }

  std::vector<double> matrix_a;
  std::vector<double> matrix_b;

  // матрицу B получают все процессы
  matrix_b = std::get<4>(in);

  // матрицу А получит полностью только 0й процесс
  if (proc_rank == 0) {
    matrix_a = std::get<3>(in);
  }

  // если всего 1 процесс - последовательное умножение
  if (process_n == 1) {
    GetOutput() = SeqMatrixMult(m, n, k, matrix_a, matrix_b);
    return true;
  }

  // какие строки каждый процесс будет перемножать
  // [start0, end0, start1, end1, ...]
  std::vector<int> ranges;

  if (proc_rank == 0) {
    auto proc_n_sizet = static_cast<size_t>(process_n);
    ranges.resize(proc_n_sizet * 2);

    // минимум на обработку
    int base = m / process_n;
    // остаток распределим
    int remain = m % process_n;

    int start = 0;
    for (size_t i = 0; i < proc_n_sizet; i++) {
      int part = base;
      if (remain > 0) {
        part++;
        remain--;
      }

      // i+i ~~ size_t+size_t ....  i*2 ~~ size_t*int (clang-tidy: size_t*int - is bad)
      ranges[i + i] = start;
      ranges[i + i + 1] = start + part;  // end (не включительно)

      start += part;
    }
  }

  // my_range получит [start, end]
  std::vector<int> my_range = {0, 0};

  // local_matrix — локальный блок матрицы A данного процесса
  std::vector<double> local_matrix;

  // отправляю всем часть матрицы A в local_matrix
  SendData(k, proc_rank, process_n, my_range, ranges, local_matrix, matrix_a);

  // теперь каждый владеет своим куском (local_matrix)
  // вызываем обычное перемножение, результат умножение кладется в local_matrix
  MatrixPartMult(k, n, local_matrix, matrix_b);

  // сбор результатов назад к 0му
  if (proc_rank == 0) {
    std::vector<double> final_result(static_cast<size_t>(m * n));
    int my_rows = ranges[1] - ranges[0];  // сколько строк
    // копирую        откуда          до куда                 куда
    std::ranges::copy(local_matrix.begin(), local_matrix.end(), final_result.begin());

    // смещение; изначально равно количеству уже записанных значений
    int offset = my_rows * n;
    auto proc_n_sizet = static_cast<size_t>(process_n);
    for (size_t i = 1; i < proc_n_sizet; ++i) {
      int start_i = ranges[i + i];
      int end_i = ranges[i + i + 1];
      int rows = end_i - start_i;

      // сколько в данной пачке элементов
      int count = rows * n;

      MPI_Recv(final_result.data() + static_cast<size_t>(offset), count, MPI_DOUBLE, static_cast<int>(i), 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      offset += count;
    }
    // 0й процесс собрал всё от других процессов

    GetOutput() = final_result;

  } else {
    // другие процессы посылают свои результаты основному 0му процессу
    int rows = my_range[1] - my_range[0];
    MPI_Send(local_matrix.data(), rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

// ===============================
// ==== дополнительные функции ===
// ===============================

void VotincevDMatrixMultMPI::SendData(int k, int proc_rank, int process_n, std::vector<int> &my_range,
                                      std::vector<int> &ranges, std::vector<double> &local_matrix,
                                      std::vector<double> &matrix_a) {
  if (proc_rank == 0) {
    // заполняю данные для себя — свои строки матрицы А
    int my_start = ranges[0];
    int my_end = ranges[1];
    int my_rows = my_end - my_start;

    auto local_matrix_size = static_cast<size_t>(my_rows) * static_cast<size_t>(k);
    local_matrix.resize(local_matrix_size);
    for (size_t i = 0; i < local_matrix_size; i++) {
      local_matrix[i] = matrix_a[i];
    }

    // рассылаю остальным
    for (int i = 1; i < process_n; i++) {
      int start_i = ranges[i + i];
      int end_i = ranges[i + i + 1];

      MPI_Send(&start_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&end_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

      int elem_count = (end_i - start_i) * k;

      auto offset = static_cast<size_t>(start_i) * static_cast<size_t>(k);
      MPI_Send(matrix_a.data() + offset, elem_count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }

  } else {
    // получаю диапазон
    MPI_Recv(my_range.data(), 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&my_range[1], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start_i = my_range[0];
    int end_i = my_range[1];
    int elem_count = (end_i - start_i) * k;

    local_matrix.resize(elem_count);

    // получаю матрицу (часть)
    MPI_Recv(local_matrix.data(), elem_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

// простое последовательное умножение (если кол-во_процессов == 1)
std::vector<double> VotincevDMatrixMultMPI::SeqMatrixMult(int param_m, int param_n, int param_k,
                                                          std::vector<double> &matrix_a,
                                                          std::vector<double> &matrix_b) {
  std::vector<double> matrix_res;
  matrix_res.assign(static_cast<size_t>(param_m) * static_cast<size_t>(param_n), 0.0);

  for (int i = 0; i < param_m; ++i) {
    for (int j = 0; j < param_n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < param_k; ++k) {
        sum += matrix_a[(i * param_k) + k] * matrix_b[(k * param_n) + j];
      }
      matrix_res[(i * param_n) + j] = sum;
    }
  }
  return matrix_res;
}

// умножение части матрицы A на всю матрицу B
void VotincevDMatrixMultMPI::MatrixPartMult(int param_k, int param_n, std::vector<double> &local_matrix,
                                            const std::vector<double> &matrix_b) {
  size_t str_count = local_matrix.size() / param_k;

  std::vector<double> result;
  result.resize(str_count * param_n);

  for (size_t i = 0; i < str_count; i++) {
    for (int j = 0; j < param_n; j++) {
      double sum = 0.0;
      for (int k = 0; k < param_k; k++) {
        sum += local_matrix[(i * param_k) + k] * matrix_b[(k * param_n) + j];
      }
      result[(i * param_n) + j] = sum;
    }
  }
  local_matrix = result;
}

bool VotincevDMatrixMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_matrix_mult
