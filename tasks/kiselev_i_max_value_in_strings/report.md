# Отчет по реализации алгоритма: находение максимального элемента в каждой строке матрицы

- **Студент**: Киселев Игорь Вячеславович, группа 3823Б1ФИ1
- **Преподаватель:**: Нестеров Александр Юрьевич и Оболенский Арсений Андреевич
- **Технология**: SEQ | MPI 
- **Вариант**: 15

## 1. Введение
Цель работы - исследовать алгоритм нахождения максимального элемента в каждой строке матрицы и оценить эффективность его последовательной (SEQ) и параллельной (MPI) реализаций.
Обработка двумерных данных является одной из базовых задач при анализе информации, работе с изображениями, моделировании и научных вычислениях. Определение максимальных значений по строкам используется для предварительной обработки, выделения характерных признаков и других целей.
В условиях роста объёмов данных (к примеру, в области машинного обучения) возрастает необходимость в ускорении вычислений, что делает сравнение последовательных и параллельных подходов актуальным. Данная работа направлена на исследование особенностей этих реализаций и оценку их производительности на различных входных данных. 

## 2. Постановка задачи.
Требуется реализовать алгоритм, который для заданной матрицы целых чисел определяет максимальный элемент в каждой её строке.
**Входные данные:**
На вход подается матрица вида ```std::vector<std::vector<int>>```, где каждый вложенный вектор представляет одну конкретную строку.
  Матрица может быть задана как: 
- Матрица прямоугольной\квадратной и рваной формы
- Пустая матрица
- Матрица с пустыми строками.
**Выходные данные:**
Вектор ```std::vector<int>```, содержащий максимальные элементы каждой строки. Для пустой матрицы предусмотрен вывод пустого вектора, для матрицы с пустыми строками тоже предусмотрена особая обработка.
  
**Ограничения и условия:**
- Значения элементов — произвольные целые числа.
- Размер матрицы не ограничен, но должен корректно обрабатываться.
- Матрица может содержать строки разной длины и\или пустые строки. Реализация SEQ и MPI версий должна корректно обрабатывать такие случаи.
- Параллельная реализация (MPI) должна выполнять распределение строк между процессами, обеспечить корректную передачу данных, вычисление максимумов и сбор результатов.

## 3. Базовый алгоритм (SEQ)
**Последовательный алгоритм выполняет:**
1. Инициализацию пустого результирующего вектора.
2. Проход по всем строкам матрицы.
3. Для каждой строки:
	- если строка пустая, то добавить корректное значение 
	- иначе, найти максимум с помощью std::max_element.
4. Вернуть результирующий вектор.
  
**Ключевые особенности реализации**
Алгоритм работает с любой целочисленной матрицей.
Вычисления выполняются строго последовательно.
Поддержка исключительных случаев встроена в основную логику без выброса ошибок.
**Псевдокод**
```
for each row in matrix:
    if row.empty():
        result.push_back(INT_MIN)
    else:
        result.push_back(max(row))
return result
```
Сложность: `O(Σ длины строк)`.
  
## 4. Схема распаралелливания(MPI)
### Распределение данных
1. Главный процесс (rank 0):
	- преобразует матрицу в плоский массив чисел;
	- вычисляет количество строк, выделяемое каждому процессу; 
	- формирует массивы:
	- *sendcounts* — количество элементов каждой подматрицы,
	- *displs* — смещения в плоской матрице.
2. Процессы получают свою часть строк с помощью *MPI_Scatterv*.

### Обработка

**Каждый процесс:**
- получает свои строки;
- для каждой строки вычисляет максимум;
- отправляет локальный результат обратно на *rank 0*.
### Сбор результатов
*Rank 0* собирает результаты с помощью *MPI_Gatherv*.

## 5. Детали реализации
### Структура проекта:
  
- `kiselev_i_max_value_in_strings\mpi\src\ops_mpi.cpp` и `kiselev_i_max_value_in_strings\mpi\include\ops_mpi.hpp` - реализация с использованием MPI.
- `tasks\kiselev_i_max_value_in_strings\seq\src\ops_seq.cpp` и `tasks\kiselev_i_max_value_in_strings\seq\include\ops_seq.hpp` — реализация последовательной версии(SEQ).
- `tasks\kiselev_i_max_value_in_strings\common\include\common.hpp` — общие типы входных/выходных данных и интерфейсы.
- `kiselev_i_max_value_in_strings\tests\functional\main.cpp` и `kiselev_i_max_value_in_strings\tests\performance\main.cpp` - набор функциональных и перфоманс тестов, для покрытия кода и проверки.

### Реализационные детали
**Обработка пустой и рваной матрицы:**
В обоих вариантах (SEQ и MPI) предусмотрена корректная обработка:
	- полностью пустой матрицы (результат — пустой вектор);
	- матрицы с пустыми строками (результат — в соответствующей позиции возвращается `std::numeric_limits<int>::min()` 
	- рваной матрицы (строки разной длины). MPI-алгоритм при этом корректно передает длины строк каждому процессу, исключая выход за границы буфера.
  
**Трансформация матрицы в плоский массив (MPI)**
Корневой процесс преобразует двумерный вектор в одну плоскую область памяти, а также формирует два вспомогательных массива:
	- `row_sizes` — количество элементов в каждой строке;
	- `displs` — смещения каждой строки в плоском массиве.
Это позволяет передавать процессы ровно те части данных, которые им нужны, без лишних пустых элементов.
  
**Распределение строк между процессами (MPI)**
Распределение производится по схеме:
	- каждая строка целиком принадлежит одному процессу;
	- строки распределяются равномерно, с разницей не более одной 		- строки между процессами;
	- каждому процессу передаются:
		1. количество строк,
		2. размеры каждой строки,
		3. сами данные строк (в плоском формате).
Такой подход исключает необходимость пересчитывать глобальные индексы внутри строки и упрощает логику поиска максимума.

**Локальный расчет максимума (MPI)**
Каждый процесс обрабатывает только свой набор строк.
Для каждой строки:
	- если строка пустая — возвращается минимально возможное предопределённое значение;
	- иначе производится линейный проход по элементам и выбор максимума.
Этот этап выполняется полностью независимо, что обеспечивает отсутствие синхронизаций между процессами.

**Сбор результатов**
Корневой процесс выполняет `MPI_Gatherv`, собирая по одному числу от каждого процесса (максимум каждой строки).
Использование `Gatherv` оправдано тем, что разные процессы могут обрабатывать разное количество строк.
  
**Единообразие буферов**
Для унификации логики передачи данных длины строк передаются отдельно, что снимает необходимость создавать выровненные массивы и удаляет потенциальную проблему с padding-значениями(Padding values в Gather подаются вынужденно и могут привести к неожиданным результатам, т.к. могут быть чем угодно).
Буфер каждого процесса содержит только нужные элементы, что экономит память и упрощает код.

**Память:** 
Каждый процесс хранит только свою подматрицу; плоский массив создаётся только в *rank 0* ; дополнительные буферы: *sendcounts*, *displs*.
## 6. Тестовая конфигурация
### Оборудование: 
  
-	**Процессор:** Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
-	**ОЗУ:** 8 ГБ
- **ОС:** Windows 10 Pro
- **MPI:** Microsoft MPI версии 10.1.12498.18
- **Режим компиляции:** Release с оптимизацией (`/O2`)

## 7. Корректность результатов тестов и выводу

### 7.1 Корректность
Корректность проверена с помощью функциональных тестов:
Сравнения MPI результата с SEQ эталоном.
Набора модульных тестов:
	- пустая матрица,
	- матрица с пустыми строками,
	- рваная матрица,
	- большие наборы данных.
  - наборы данных
Все тесты пройдены успешно.

### 7.2 Перфоманс тесты
Первоманс тесты в режиме дебаг.
Исходная матрица содержит набор из n строк, а длина каждой строки увеличвается в зависимости от её индекса в. То есть имеем рваную матрицу. 
  
 | Размер данных | Режим        | Время, s   | Ускорение (S = Tseq / Tmpi) 
| ------------- | ------------ | ---------- | ---------------------------- 
| **5000**      | SEQ          | 0.01306464 | 1.00                         
|               | MPI (4 proc) | 0.04009702 | 0.33                         
| **10000**     | SEQ          | 0.06652354 | 1.00                         
|               | MPI (4 proc) | 0.16494808 | 0.40                         
| **20000**     | SEQ          | 0.32634116 | 1.00                         
|               | MPI (4 proc) | 0.83355846 | 0.39                         
| **30000**     | SEQ          | 2.89780174 | 1.00                         
|               | MPI (4 proc) | 3.17981036 | 0.91                         
  
MPI менее эффективен на малых матрицах из-за накладных расходов на распределение и сбор данных. Эффективность растёт с увеличением объёма данных.

## 8. Заключение

В работе реализовано и исследовано последовательное и MPI-распараллеленное решение задачи нахождения максимума по строкам матрицы.
MPI-версия при малых и средних объёмах данных оказывается менее эффективной из-за накладных расходов на коммуникацию и распределение данных. Эффект ускорения MPI заметен только на больших матрицах (30000 строк и более), однако низкие характеристики тестового компьютера ограничивают возможность проведения экспериментов с ещё большими объёмами данных..
Основным ограничивающим фактором являются 
  - стоимость коммуникаций MPI
  - неоднородность размеров строк
  - накладные расходы на раздачу и сбор данных.
При этом MPI-решение экономично расходует память: каждый процесс получает только необходимые данные, без создания полной копии исходной матрицы для всех процессов. Это обеспечивает масштабируемость алгоритма и позволяет обрабатывать большие объёмы данных без лишних затрат памяти.
Решение корректно обрабатывает пустые строки и рваные матрицы.
Код прошёл все модульные тесты и соответствует требованиям  для включения в библиотеку PPC.

## 9. Литература
1. https://learning-process.github.io/parallel_programming_course/ru/
2. https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-scatterv-function
3. https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-bcast-function
4. https://habr.com/ru/articles/121235/
5. https://disk.yandex.ru/d/NvHFyhOJCQU65w
  
## Приложения (код параллельной реализации)
```
void KiselevITestTaskMPI::DistributeRowLengths(const std::vector<std::vector<int>> &matrix, int total_rows,
                                               int world_rank, int world_size, std::vector<int> &local_row_lengths,
                                               std::vector<int> &len_counts, std::vector<int> &len_displs) {
  std::vector<int> all_row_lengths;
  int base = total_rows / world_size;
  int rem = total_rows % world_size;

  if (world_rank == 0) {
    all_row_lengths.resize(static_cast<size_t>(total_rows));
    for (int i = 0; i < total_rows; ++i) {
      all_row_lengths[static_cast<size_t>(i)] = static_cast<int>(matrix[static_cast<size_t>(i)].size());
    }

    int offset = 0;
    for (int pr = 0; pr < world_size; ++pr) {
      len_counts[pr] = base + (pr < rem ? 1 : 0);
      len_displs[pr] = offset;
      offset += len_counts[pr];
    }
  }

  MPI_Scatterv(all_row_lengths.data(), len_counts.data(), len_displs.data(), MPI_INT, local_row_lengths.data(),
               static_cast<int>(local_row_lengths.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

void KiselevITestTaskMPI::DistributeValues(const std::vector<std::vector<int>> &matrix, int world_rank, int world_size,
                                           const std::vector<int> &len_counts, const std::vector<int> &len_displs,
                                           std::vector<int> &local_values) {
  std::vector<int> val_counts(world_size);
  std::vector<int> val_displs(world_size);

  if (world_rank == 0) {
    int offset = 0;
    for (int pr = 0; pr < world_size; ++pr) {
      int count = 0;
      for (int i = 0; i < len_counts[pr]; ++i) {
        count += static_cast<int>(matrix[len_displs[pr] + i].size());
      }
      val_counts[pr] = count;
      val_displs[pr] = offset;
      offset += count;
    }
  }

  MPI_Bcast(val_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(val_displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    int total_elements = val_displs[world_size - 1] + val_counts[world_size - 1];
    flat_matrix.reserve(static_cast<size_t>(total_elements));
    for (const auto &row : matrix) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  }

  int my_count = val_counts[world_rank];
  if (my_count > 0) {
    local_values.resize(my_count);
  } else {
    local_values.clear();
  }

  MPI_Scatterv(flat_matrix.data(), val_counts.data(), val_displs.data(), MPI_INT, local_values.data(), my_count,
               MPI_INT, 0, MPI_COMM_WORLD);
}

void KiselevITestTaskMPI::ComputeLocalMax(const std::vector<int> &local_values,
                                          const std::vector<int> &local_row_lengths, std::vector<int> &local_result) {
  size_t n_rows = local_row_lengths.size();
  if (n_rows == 0) {
    return;
  }

  local_result.resize(n_rows);
  int pos = 0;

  for (size_t rw = 0; rw < n_rows; ++rw) {
    int len = local_row_lengths[rw];
    if (len == 0) {
      // пустая строка → используем минимальное значение int
      local_result[rw] = std::numeric_limits<int>::min();
    } else {
      int tmp_max = local_values[pos];
      for (int j = 1; j < len; ++j) {
        tmp_max = std::max(tmp_max, local_values[pos + j]);
      }
      local_result[rw] = tmp_max;
    }
    pos += len;
  }
}

bool KiselevITestTaskMPI::RunImpl() {
  const auto &matrix = GetInput();
  auto &result_vector = GetOutput();

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_rows = static_cast<int>(matrix.size());
  MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base = total_rows / world_size;
  int rem = total_rows % world_size;
  int local_row_count = base + (world_rank < rem ? 1 : 0);

  std::vector<int> local_row_lengths(local_row_count);
  std::vector<int> len_counts(world_size);
  std::vector<int> len_displs(world_size);
  DistributeRowLengths(matrix, total_rows, world_rank, world_size, local_row_lengths, len_counts, len_displs);

  std::vector<int> local_values;
  DistributeValues(matrix, world_rank, world_size, len_counts, len_displs, local_values);

  std::vector<int> local_result;
  ComputeLocalMax(local_values, local_row_lengths, local_result);

  if (world_rank == 0) {
    result_vector.resize(total_rows);
  }

  MPI_Gatherv(local_result.data(), local_row_count, MPI_INT, result_vector.data(), len_counts.data(), len_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (total_rows > 0) {
    MPI_Bcast(result_vector.data(), total_rows, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}
```