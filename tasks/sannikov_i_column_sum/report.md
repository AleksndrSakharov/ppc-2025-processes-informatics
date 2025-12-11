# Сумма значений по столбцам матрицы

- Студент: Санников Иван Михайлович, Группа: 3823Б1ФИ2
- Технология: SEQ, MPI
- Вариант: 12
## 1. Введение
Вычисление сумм по столбцам матрицы является одной из основных в математике и часто используется в прикладных задачах, как машинное обучение(ML) и статистика. Часто приходится работать с большим объемом данных, что делает задачу ресурсоемкой при использовании последовательных алгоритмов. Цель данной лабораторной работы - написание алгоритма на основе технологии MPI для распределение нагрузки между несколькоми процессами. 

## 2. Постановка задачи
Входные данные: std::vector<std::vector<int>> - вектор векторов типа данных int представляющий из себя матрицу.

Для матрицы размера A*B, строица std::vector<int> размера B, где для элемента i хранится сумма столбца i из входной матрицы.

## 3. Базовый алгоритм(seq)
Последовательный алгорритм:
- Проходим по всем строкам
- В каждой i строке проходим по всем столбцам
- Складываем элемент i строки j столбца c элементом j вектора суммы.

```cpp
bool SannikovIColumnSumSEQ::RunImpl() {
  const auto &input_matrix = GetInput();
  for (const auto &row : input_matrix) {
    std::size_t column = 0;
    for (const auto &value : row) {
      GetOutput()[column] += value;
      column++;
    }
  }
  return !GetOutput().empty();
}
```

## 4. Описание параллельного алгоритма 

1. Обрабатываем входные данные: Превращаем std::vector<std::vector<int>> в последовательный std::vector<int>, чтобы была возможность его разделить Scatterv.
2. Вычисляем сколько данных получит процесс: перемножаем количество столбцов на строки, делим на количество процессов и прибавляем 1, если номер вычисления меньше остатка (i<rem?1:0). Получив массив элементов сдвига вычисляем сам сдвиг для определенного rank. Сдвиг равен элементу массива сдвига по номеру rank и вычисляем его остаток от деления на количество столбцов (id_elem[rank] % columns).
3. Рассылаем данные MPI_Scatterv.
4. Вычисляем локальные суммы: Создается локальный вектор сум. При помощи вычисленных сдвигов заполняется значениями сум столбцов входной матрицы.
5. Собираются все данные при помощи MPI_Allreduce.

## 5. Experimental Setup

- Hardware/OS: Intel i9 13900KF, 24 ядра, RAM: 32Gb, OS: Windows 11
- Toolchain: Cmake 3.28.3, g++ (Ubuntu 14.2.0 x86_64), Docker-контейнер, Режим сборки: Release.
- Data: Для тестов на производительность использовалась матрица размером 10000 на 10000 заполненнаная по алгоритму: элемент i строки j столбца равен i * 14 + j * 21.

## 6. Результаты

### 6.1 Корректность 

Корректность работы алгоритма проверена при помощи технологии Google Test. Для проверки использовались входные данные в виде матриц разных размеров, пустые матрицы и матрицы с отрицательными числами. 

### 6.2 Производительность

Входные данные: Матрица 10000 на 10000.

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.020   | 1.00    | N/A        |
| mpi         | 2     | 0.218   | 0.09    | 4.59%      |
| mpi         | 4     | 0.160   | 0.12    | 3.10%      |
| mpi         | 8     | 0.150   | 0.12    | 1.60%      |

## 7. Выводы

Из результатов тестов на производительность можно сделать вывод, что данная задача не подходит для реализации на технологии MPI. Затраты системы на обработку вызовов Scatterv и Allreduce прривышают суммарные затраты на обработку данных в последовательном алгоритме. 

## 8. Литература
1. Open MPI: Documentation - https://www.open-mpi.org/doc/
2. Александер Сысоев. Курс лекций по параллельному программированию. Лекция № 3 - https://cloud.unn.ru/s/o5y2jGbxb7XpBJa
3. Parallel Programming 2025-2026 - https://disk.yandex.ru/d/NvHFyhOJCQU65w

## 9. Приложение

```cpp

void SannikovIColumnSumMPI::PrepareSendBuffer(const InType &input_matrix, int rank, std::uint64_t rows,
                                              std::uint64_t columns, std::vector<int> &sendbuf) {
  if (rank != 0) {
    return;
  }
  if (rank == 0) {
    const std::uint64_t base = rows * columns;
    sendbuf.resize(static_cast<std::size_t>(base));
    for (std::uint64_t i = 0; i < rows; i++) {
      for (std::uint64_t j = 0; j < columns; j++) {
        sendbuf[static_cast<std::size_t>((i * columns) + (j))] =
            input_matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      }
    }
  }
}



bool SannikovIColumnSumMPI::RunImpl() {
  const auto &input_matrix = GetInput();

  int rank = 0;
  int size = 1;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    rows = static_cast<std::uint64_t>(input_matrix.size());
    columns = static_cast<std::uint64_t>(input_matrix.front().size());
  }

  MPI_Bcast(&rows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  const std::uint64_t base = rows * columns;
  if (columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  if (base > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  const int columns_int = static_cast<int>(columns);
  const int base_int = static_cast<int>(base);
  GetOutput().assign(static_cast<std::size_t>(columns_int), 0);

  std::vector<int> sendbuf;
  PrepareSendBuffer(input_matrix, rank, rows, columns, sendbuf);
  std::vector<int> elem_for_proc(size);
  std::vector<int> id_elem(size);
  int displacement = 0;
  for (int i = 0; i < size; i++) {
    elem_for_proc[i] = static_cast<int>(base_int / size) + (i < (base_int % size) ? 1 : 0);
    id_elem[i] = displacement;
    displacement += elem_for_proc[i];
  }
  const int mpi_displacement = id_elem[rank] % static_cast<int>(columns_int);
  std::vector<int> buf(static_cast<std::size_t>(elem_for_proc[rank]), 0);
  MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, elem_for_proc.data(), id_elem.data(), MPI_INT, buf.data(),
               elem_for_proc[rank], MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> sum(static_cast<std::size_t>(columns_int), 0);
  for (int i = 0; i < (elem_for_proc[rank]); i++) {
    int new_col = (i + mpi_displacement) % columns_int;
    sum[static_cast<std::size_t>(new_col)] += buf[static_cast<std::size_t>(i)];
  }
  MPI_Allreduce(sum.data(), GetOutput().data(), columns_int, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return !GetOutput().empty();
}


```