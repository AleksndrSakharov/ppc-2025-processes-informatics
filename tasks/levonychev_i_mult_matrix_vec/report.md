# Ленточная горизонтальная схема - умножение матрицы на вектор

- Студент: Левонычев Иван Александрович, группа 3823Б1ФИ3
- Технология: SEQ, MPI
- Вариант: 11

## 1. Введение
- Мотивация: продолжить изучать возможности и инструменты MPI, посмотреть, какой прирост производительности даст MPI версия относительно последовательной.
- Проблема: задача умножения матрицы на вектор - распространненая задача в линейной алгебре, поэтому ее оптмизация вызывает интерес.
- Ожидаемый результат: ожидается, что MPI версия будет работать быстрее последовательной за счёт распределения вычислительной нагрузки между несколькими процессами.

## 2. Постановка задачи
На вход программе подаются 4 параметра:
1. Матрица A в виде одномерного массива чисел. Тип элементов - double.
2. Количество строк (*rows*).
3. Количество столбцов (*cols*).
4. Вектор x в виде одномерного массива чисел. Тип элементов - double.
Требуется вычислить вектор b, который является произведением матрицы A и вектора x:    $A*x = b$.

## 3. Базовый алгоритм (последовательный)
Последовательный алгоритм довольно прост. Мы проходим в цикле по строкам матрицы. Затем с помощью внутреннего цикла накапливаем результат - то есть ищем скалярное произведение i-ой строки матрицы и вектора x. Данное скалярное произведение будет i-ой координатой вектора b.
Код функции RunImpl:
```cpp
bool LevonychevIMultMatrixVecSEQ::RunImpl() {
  const std::vector<double> &matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  const std::vector<double> &vec_x = std::get<3>(GetInput());

  OutType &result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    double scalar_product = 0;
    for (int j = 0; j < cols; ++j) {
      scalar_product += matrix[(i * cols) + j] * vec_x[j];
    }
    result[i] = scalar_product;
  }
  return true;
}
```

## 4. Описание параллельного алгоритма
Параллельный алгоритм основан на последовательной версии, единственное различие в том, что каждый процесс считает свою часть итогового вектора b. Каждому процессу достается $\frac{N}{P}$ строк матрицы A, где N - общее количество строк, P - количество процессов. Распределение происходит с помощью функции Scatterv. Затем каждый процесс умножает свою часть матрицы A на полный вектор x (алгоритм перемножения идентичен последовательной версии), тем самым получая свою часть вектора b. Последний шаг - сбор всех результатов на всех процессах с помощью функции Allgatherv.

## 5. Experimental Setup
- Hardware/OS: Intel i5-12450H, 8 ядер, RAM 16GB, Windows 11
- Toolchain: Microsoft Visual C++ (MSVC), Release
- Data: Матрица размера 4096 на 4096.

## 6. Результаты

### 6.1 Корректность
Коррекность последовательного и MPI алгоритмов проверена функциональными тестами в количестве 8 штук. Тестировалось на матрицах разных размерностей, в том числе содержащих 1 строку и/или 1 столбец.

### 6.2 Производительность
Входные данные: Матрица размера 4096 на 4096.

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.011   | 1.00    | N/A        |
| mpi         | 2     | 0.035   | 0.31    | 15.7%      |
| mpi         | 4     | 0.027   | 0.40    | 10.0%      |

Можем видеть, что параллельный алгоритм не дает прироста производительности. Это происходит из-за того, что бОльшая часть времени уходит на накладные расходы, связанные с созданием локальной матрицы и работой функции Scatterv. Эксперименты показали, что только 50% всего времени занимает работа функции Scatterv, еще 15% на выделение памяти для локальной матрицы. Если замерять только лишь сами вычисления, то прирост производительности ожидаемый - в P раз, где P - количество процессов.

## 7. Выводы
На основе результатов производительности можно сделать вывод, что параллельный алгоритм, решающий данную задачу, имеет смысл, если только оптимизировать копирование общих данных в локальные переменные.

## 8. Литература
1. Стандарт MPI.
2. Лекции и практики по параллельному программированию.

## 9. Приложение

```cpp
bool LevonychevIMultMatrixVecMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  OutType &global_b = GetOutput();

  int rows = 0;
  int cols = 0;
  std::vector<double> x;

  std::vector<int> recvcounts_scatterv(proc_num);
  std::vector<int> displs_scatterv(proc_num);
  std::vector<int> recvcounts_gatherv(proc_num);
  std::vector<int> displs_gatherv(proc_num);

  rows = std::get<1>(GetInput());
  cols = std::get<2>(GetInput());
  x = std::get<3>(GetInput());
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  x.resize(cols);
  MPI_Bcast(x.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for (int i = 0; i < proc_num; ++i) {
    int local_count_of_rows = i == (proc_num - 1) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
    recvcounts_scatterv[i] = local_count_of_rows * cols;
    recvcounts_gatherv[i] = local_count_of_rows;
    int start = i * (rows / proc_num);
    displs_scatterv[i] = start * cols;
    displs_gatherv[i] = start;
  }

  int local_count_of_rows = (proc_rank == (proc_num - 1)) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
  int recvcount = local_count_of_rows * cols;
  OutType local_matrix(recvcount);
  MPI_Scatterv(std::get<0>(GetInput()).data(), recvcounts_scatterv.data(), displs_scatterv.data(), MPI_DOUBLE,
               local_matrix.data(), recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  OutType local_b(local_count_of_rows);
  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start = cols * i;
    double scalar_product = 0;
    for (int j = 0; j < cols; ++j) {
      scalar_product += local_matrix[start + j] * x[j];
    }
    local_b[i] = scalar_product;
  }

  MPI_Allgatherv(local_b.data(), local_count_of_rows, MPI_DOUBLE, global_b.data(), recvcounts_gatherv.data(),
                 displs_gatherv.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  return true;
}
```