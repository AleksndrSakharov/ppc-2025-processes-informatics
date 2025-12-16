#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

VotincevDAlternatingValuesMPI::VotincevDAlternatingValuesMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

// проверка входных данных не нужна - пустой вектор тоже обрабатывается
bool VotincevDAlternatingValuesMPI::ValidationImpl() {
  return true;
}

// нет ничего в PreProcessing
bool VotincevDAlternatingValuesMPI::PreProcessingImpl() {
  return true;
}

// код MPI
bool VotincevDAlternatingValuesMPI::RunImpl() {
  // получаю кол-во процессов
  int process_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_n);

  // получаю ранг процесса
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // вектор значений (его заполняет только 0й процесс)
  std::vector<double> vect_data;

  if (proc_rank != 0) {
    GetOutput() = -1;  // специальное значение
  }

  // 0й процесс получает данные
  if (proc_rank == 0) {
    vect_data = GetInput();

    // если процессов больше, чем размер вектора — уменьшаем
    // например: вектор = 2 элемента, а процессов запущено 4
    process_n = std::min<int>(process_n, static_cast<int>(vect_data.size()));
  }

  // рассылаю всем "актуальное" число процессов
  // т.к. его мог уменьшить процесс 0
  MPI_Bcast(&process_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // если вектор пустой
  if (process_n == 0) {
    return true;
  }

  if (proc_rank == 0) {
    GetOutput() = ProcessMaster(process_n, vect_data);  // работа 0-го процесса
  } else if (proc_rank < process_n) {
    ProcessWorker();  // работа процессов-работников
  }

  return true;
}

bool VotincevDAlternatingValuesMPI::PostProcessingImpl() {
  return true;
}

// ============ дополнительные функции ============

// вспомогательная, if с такими условиями выглядит страшно
bool VotincevDAlternatingValuesMPI::IsSignChange(const double &a, const double &b) {
  return ((a < 0 && b >= 0) || (a >= 0 && b < 0));
}

// работа 0-го процесса (главного)
int VotincevDAlternatingValuesMPI::ProcessMaster(int process_n, const std::vector<double> &vect_data) {
  int all_swaps = 0;
  const int n = static_cast<int>(vect_data.size());

  // делим на части
  const int base = n / process_n;  // минимум на обработку
  int remain = n % process_n;      // остаток (распределяем между первыми)

  int start_id = 0;

  // распределяем работу между процессами-рабочими
  for (int worker = 1; worker < process_n; worker++) {
    int part_size = base;

    // если остались "лишние" элементы — раздаём по одному первым процессам
    if (remain > 0) {
      part_size++;
      remain--;
    }

    // каждому процессу (кроме последнего) даём ещё +1 элемент справа,
    // чтобы проверить возможное чередование между границами блоков
    part_size++;

    // отправляем размер части
    MPI_Send(&part_size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);

    // отправляем сами данные
    MPI_Send(vect_data.data() + start_id, part_size, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);

    // сдвигаем начало следующего куска
    start_id += part_size - 1;  // -1 потому что правый сосед был добавлен
  }

  // 0й процесс считает пересесчения в своей части
  int part_size = n - start_id;
  for (int i = start_id + 1; i < start_id + part_size; i++) {
    if (IsSignChange(vect_data[i - 1], vect_data[i])) {
      all_swaps++;
    }
  }

  // собираем ответы от всех рабочих процессов
  for (int worker = 1; worker < process_n; worker++) {
    int tmp = 0;
    MPI_Recv(&tmp, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    all_swaps += tmp;
  }

  return all_swaps;
}

// работа процессов 1,2, ...
void VotincevDAlternatingValuesMPI::ProcessWorker() {
  int part_size = 0;

  // получаю, сколько элементов прислал процесс 0
  MPI_Recv(&part_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // выделяю память под кусок
  std::vector<double> data(part_size);

  // получаю свою часть вектора
  MPI_Recv(data.data(), part_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int swap_count = 0;

  // считаю чередования знаков
  for (int i = 1; i < part_size; i++) {
    if (IsSignChange(data[i - 1], data[i])) {
      swap_count++;
    }
  }

  // отправляю результат назад 0-му процессу
  MPI_Send(&swap_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

}  // namespace votincev_d_alternating_values
