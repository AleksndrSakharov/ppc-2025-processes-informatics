# Нахождение минимальных значений по строкам матрицы

- Студент: Левонычев Иван Александрович, группа 3823Б1ФИ3
- Технология: SEQ, MPI
- Вариант: 17

## 1. Введение
- Мотивация: изучить базовые возможности и инструменты MPI на простой задаче, посмотреть, какой прирост производительности даст MPI версия относительно последовательной.
- Проблема: задача нахождения минимальных значений по строкам матрицы - распространненая задача при работе с матрицами, поэтому ее оптмизация вызывает интерес.
- Ожидаемый результат: ожидается, что MPI версия будет работать быстрее последовательной за счёт распределения вычислительной нагрузки между несколькими процессами.

## 2. Постановка задачи
На вход программе подаются 3 параметра:
1. Матрица в виде одномерного массива чисел. Тип элементов - int.
2. Количество строк (*rows*).
3. Количество столбцов (*cols*).
Требуется найти минимумы по строкам матрицы, то есть на выходе должен быть одномерный массив чисел длины *rows*, в котором на i-ой позиции находится минимум из i-ой строки исходной матрицы.

## 3. Базовый алгоритм (последовательный)
Последовательный алгоритм довольно прост. Мы проходим в цикле по строкам матрицы. Инициализируем изначальный минимум первым элементов строки, а затем каждый раз находим минимум из текущего минимума и следующего элемента строки и присваиваем в переменную, содержащую текущий минимум.
Код функции RunImpl:
```cpp
bool LevonychevIMinValRowsMatrixSEQ::RunImpl() {
  const std::vector<int> &matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  OutType &result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    int min_val = matrix[static_cast<size_t>(cols) * static_cast<size_t>(i)];
    for (int j = 1; j < cols; ++j) {
      min_val = std::min(matrix[(cols * i) + j], min_val);
    }
    result[i] = min_val;
  }

  return true;
}
```

## 4. Описание параллельного алгоритма
Параллельный алгоритм основан на последовательной версии, единственное различие в том, что каждый процесс считает минимумы в своих строках.
Распределение строк происходит с помощью функции Scatterv по следующему принципу:
1. Если количество строк меньше количества процессов, то все строки отдаются процессу с рангом 0, остальные процессы ничего не делают.
2. Если количество строк нацело делится на количество процессов, то всем процессам достается равное количество строк. Например, матрица содержит 90 строк, и у нас 3 процесса. То процессу с рангом 0 достанутся строки 0-29, процессу с рангом 1 строки 30-59, и процессу с рангом 2 строки 60-89.
3. Если количество строк не делится нацело на количество процессов, то последнему процессу добавляются оставшиеся строки. Например, матрица содержит 100 строк, и у нас 3 процесса. Тогда процессу с рангом 0 достанутся строки 0-29, процессу с рангом 1 строки 30-59, а процессу с рангом 2 строки 60-99.

Каждый процесс считает свои минимумы, затем с помощью функции GatherV все минимумы объдиняются в общий массив на процессе с рангом 0. Потом процесс с рангом 0 рассылает этот массив всем оставшимся процессам с помощью функции MPI_Bcast.

## 5. Experimental Setup
- Hardware/OS: Intel i5-12450H, 8 ядер, RAM 16GB, Windows 11
- Toolchain: Microsoft Visual C++ (MSVC), Release
- Data: Матрица размера 1024 на 262144, заполнение начинается от 0 с шагом 1.

## 6. Результаты

### 6.1 Корректность
Коррекность последовательного и MPI алгоритмов проверена функциональными тестами в количестве 10 штук. Тестировалось на матрицах разных размерностей, в том числе содержащих 1 строку и/или 1 столбец.

### 6.2 Производительность
Входные данные: Матрица размера 1024 на 262144.

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.165   | 1.00    | N/A        |
| mpi         | 2     | 0.325   | 0.51    | 25.4%      |
| mpi         | 4     | 0.225   | 0.73    | 18.3%      |

Можем видеть, что параллельный алгоритм не дает прироста производительности. Это происходит из-за того, что бОльшая часть времени уходит на накладные расходы, связанные с созданием локальной матрицы и работой функции Scatterv. Эксперименты показали, что на двух процессах только лишь на создание локальной матрицы и работу Scatterv уходит примерно 0.2 секунды, а общее время выполнения RunImpl() - 0.32 секунды.

## 7. Выводы
На основе результатов производительности можно сделать вывод, что параллельный алгоритм, решающий данную задачу, имеет смысл, если только оптимизировать копирование общих данных в локальные переменные.

## 8. Литература
1. Стандарт MPI.
2. Лекции и практики по параллельному программированию.

## 9. Приложение

```cpp
bool LevonychevIMinValRowsMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  int rows = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    rows = std::get<1>(GetInput());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput().resize(rows);
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  OutType &global_min_values = GetOutput();

  int rows = 0;
  int cols = 0;
  std::vector<int> recvcounts_scatterv(proc_num);
  std::vector<int> displs_scatterv(proc_num);
  std::vector<int> recvcounts_gatherv(proc_num);
  std::vector<int> displs_gatherv(proc_num);

  if (proc_rank == 0) {
    rows = std::get<1>(GetInput());
    cols = std::get<2>(GetInput());

    for (int i = 0; i < proc_num; ++i) {
      int local_count_of_rows = i == (proc_num - 1) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
      recvcounts_scatterv[i] = local_count_of_rows * cols;
      recvcounts_gatherv[i] = local_count_of_rows;
      int start = i * (rows / proc_num);
      displs_scatterv[i] = start * cols;
      displs_gatherv[i] = start;
    }
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int local_count_of_rows = (proc_rank == (proc_num - 1)) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
  int recvcount = local_count_of_rows * cols;
  OutType local_matrix(recvcount);
  MPI_Scatterv(std::get<0>(GetInput()).data(), recvcounts_scatterv.data(), displs_scatterv.data(), MPI_INT,
               local_matrix.data(), recvcount, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_min_values(local_count_of_rows);

  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start = cols * i;
    int min_value = local_matrix[start];
    for (int j = 1; j < cols; ++j) {
      min_value = std::min(local_matrix[start + j], min_value);
    }
    local_min_values[i] = min_value;
  }

  MPI_Gatherv(local_min_values.data(), local_count_of_rows, MPI_INT, global_min_values.data(),
              recvcounts_gatherv.data(), displs_gatherv.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_min_values.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}
```
