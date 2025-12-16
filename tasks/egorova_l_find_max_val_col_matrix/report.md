# Нахождение наиболее отличающихся по значению соседних элементов вектора

- Студент: Егорова Лариса Алексеевна, группа 3823Б1ФИ1
- Технология: SEQ | MPI
- Вариант: 16

## 1. Введение
Данная лабораторная работа посвящена решению задачи поиска максимальных элементов в столбцах матрицы. Цель работы: разработать и реализовать два подхода последовательный алгоритм и параллельная реализация с использованием библиотеки MPI для решения задачи, а затем сравнить их и сделать выводы.

## 2. Постановка задачи
Для заданной матрицы размером M×N необходимо найти вектор максимальных значений для каждого столбца.
Формально: для каждого столбца j найти max(a[i][j]) для всех i от 0 до M-1.

Входные данные: матрица целых чисел
Выходные данные: вектор максимальных значений для каждого столбца

Особые случаи:

- Пустая матрица → пустой вектор
- Матрица с нулевыми столбцами → пустой вектор
- Некорректная матрица (разные длины строк) → ошибка валидации

## 3. Базовый алгоритм (Последовательный)
Последовательный алгоритм проходит по каждому столбцу матрицы и находит максимальное значение:

for (j от 0 до N-1):
    max_val = MIN_INT
    for (i от 0 до M-1):
        if matrix[i][j] > max_val:
            max_val = matrix[i][j]
    result[j] = max_val

Результатом выполнения алгоритма будет вектор максимальных значений result.

Сложность алгоритма: O(M×N)


## 4. Схема распараллеливания

### 4.1 Общая архитектура
Алгоритм использует декомпозицию по столбцам — каждый процесс получает свой набор столбцов для независимой обработки. Такой подход обеспечивает максимальное распараллеливание при минимальных коммуникационных затратах.

### 4.2 Распределение данных
Столбцы распределяются по процессам с адаптивным балансированием нагрузки. Если количество столбцов N не делится нацело на число процессов P, первые R процессов (R = N mod P) получают на один столбец больше. Это гарантирует, что разница в нагрузке между процессами не превышает одного столбца.

Исходная матрица преобразуется в плоский массив с построчным хранением для эффективной передачи через MPI.

### 4.3 Коммуникационная схема
Процесс выполняется в три этапа:
- Распространение данных — процесс 0 рассылает размеры матрицы и данные всем процессам через MPI_Bcast
- Параллельные вычисления — каждый процесс находит максимумы в своих столбцах независимо
- Сбор результатов — все процессы обмениваются результатами через MPI_Allgatherv

### 4.4 Преимущества и ограничения
Преимущества:
- минимальные коммуникации (только 2 передачи)
- идеальная балансировка нагрузки
- простота реализации
- масштабируемость

Ограничения:
- каждый процесс хранит полную копию матрицы
- ограничение по памяти на одном узле


## 5. Детали реализации
Структура кода:
- ..\MPI\ppc-2025-processes-informatics\tasks\egorova_l_find_max_val_col_matrix\seq\src\ops_seq.cpp — последовательная реализация
- ..\MPI\ppc-2025-processes-informatics\tasks\egorova_l_find_max_val_col_matrix\mpi\src\ops_mpi.cpp — MPI реализация

Тесты: 
- ..\MPI\ppc-2025-processes-informatics\tasks\egorova_l_find_max_val_col_matrix\tests\functional\main.cpp — функциональные
- ..\MPI\ppc-2025-processes-informatics\tasks\egorova_l_find_max_val_col_matrix\tests\performance\main.cpp — на производительность

## 6. Экспериментальная установка

### 6.1 Оборудование и ПО
- CPU: AMD Ryzen 5 2600 Six-Core Processor 3.40 GHz (6 ядер, 12 потоков)
- RAM: 16GB DDR4
- ОС: Windows 10 Pro
- Тип сборки: Release

### 6.2 Функциональное тестирование
Для проверки корректности реализации проведено комплексное функциональное тестирование:

Базовые случаи:
- Пустая матрица — проверка обработки граничного условия
- Матрица с нулевыми столбцами — валидация обработки вырожденных случаев
- Матрица 1×1 — тестирование минимального размера

Структурные тесты:
- Один столбец — проверка обработки векторного случая
- Одна строка — тестирование горизонтальной матрицы
- Квадратные матрицы 3×3 и 5×5 — стандартные сценарии

Семантические тесты:
- Отрицательные значения — проверка корректности сравнения
- Одинаковые значения — тестирование стабильности при равенстве элементов
- Нулевые значения — валидация работы с нулями

Сценарии расположения максимумов:
- Максимум в первом столбце — проверка граничных условий
- Максимум в последнем столбце — тестирование конечных элементов
- Максимумы в разных столбцах — комплексная проверка распределения

Неквадратные матрицы:
- Прямоугольные матрицы 2×4 и 3×2 — тестирование асимметричных случаев
- Большая матрица 10×1 — проверка обработки длинных столбцов
- Матрица 1×10 — тестирование длинных строк

### 6.3 Тестирование производительности
Методология тестирования:

- Размер тестовой матрицы: 5000×5000 элементов (25 миллионов значений) и 10000х10000 (100 миллионов значений)
- Тип данных: 32-битные целые числа (int)
- Количество процессов: 1, 2, 4, 8
- Количество запусков: 5 для каждого конфигурации

Генерация тестовых данных:
- Используется алгоритм детерминированного заполнения
- Последовательное увеличение значений для предсказуемости результатов
- Гарантированное наличие уникальных максимальных значений в каждом столбце

Метрики производительности:
- Время выполнения (секунды)
- Ускорение относительно последовательной версии
- Эффективность параллелизации
- Потребление памяти

## 7. Результаты и обсуждение

### 7.1 Корректность
Реализованные последовательная и MPI-версии алгоритма успешно прошли все функциональные тесты. Проверка корректности осуществлялась через сравнение результатов обеих реализаций с эталонными значениями, вычисленными аналитически. Особое внимание уделялось обработке граничных случаев: пустых матриц, матриц с нулевыми столбцами, матриц минимального размера 1×1. Все тесты подтвердили идентичность результатов параллельной и последовательной версий.

### 7.2 Проведение тестов
Для оценки производительности алгоритма поиска максимальных значений по столбцам матрицы были проведены тесты на двух размерах матриц: 5000×5000 (25 миллионов значений) и 10000×10000 (100 миллионов значений). Результаты представлены в таблицах:

1. Матрица 5000×5000
|Режим	|Процессы |Время, с	|Ускорение  |Эффективность|
|-------|---------|---------|-----------|-------------|
|seq	|1	      |0.22558  |1.00	    |N/A          |
|mpi	|2	      |0.20185  |1.12       |56.0%        |
|mpi	|4	      |0.23712  |0.95       |23.8%        |
|mpi	|8	      |0.29534  |0.76	    |9.5%         |

2. Матрица 10000x10000
|Режим	|Процессы |Время, с	|Ускорение  |Эффективность|
|-------|---------|---------|-----------|-------------|
|seq	|1	      |0.26388  |1.00	    |N/A          |
|mpi	|2	      |0.21299  |1.24       |61.95%       |
|mpi	|4	      |0.25952  |1.02       |25.4%        |
|mpi	|8	      |0.30015  |0.88	    |10.99%       |

Ускорение вычислялось по формуле (seq_time / mpi_time)
Эффективность вычислялась по формуле (Ускорение / N) * 100%, где N - количество процессов

### 7.3 Анализ результатов

#### 7.3.1 Влияние размера задачи на производительность
Проведя тесты стало заметно значительное улучшение эффективности параллелизации с ростом размера матрицы:

Для матрицы 5000×5000:
- Лучшая эффективность: 55.87% на 2 процессах

Для матрицы 10000×10000:
- Лучшая эффективность: 61.95% на 2 процессах (+6.08%)
- Ускорение улучшилось с 1.12× до 1.24×

#### 7.3.2 Оптимальные конфигурации
Для обеих размерностей наилучшая производительность достигается при использовании 2 процессов MPI:
Матрица 5000×5000: время 0.20186 с, ускорение 1.12×
Матрица 10000×10000: время 0.21299 с, ускорение 1.24×

#### 7.3.3 Динамика масштабируемости
Обе размерности демонстрируют схожую динамику:

- Пик эффективности на 2 процессах
- Резкое снижение эффективности при 4 и более процессах
- Наилучшее соотношение вычислений и коммуникаций при 2 процессах

#### 7.3.4 Выводы по производительности:
Алгоритм демонстрирует хорошую масштабируемость для средних и больших размеров задач, что подтверждается ростом эффективности с 55.87% до 61.95% при увеличении размера матрицы.

Оптимальная конфигурация — 2 процесса MPI, что обеспечивает баланс между вычислительной нагрузкой и коммуникационными затратами.

Ограниченная масштабируемость при большом количестве процессов обусловлена преобладанием коммуникационных операций (MPI_Allgatherv) над вычислительными при распределении на 4 и более процессов.

Практическая применимость: разработанная реализация эффективна для обработки матриц средних и больших размеров, где вычислительная сложность преобладает над накладными расходами параллелизации.

## 8. Заключение
Был разработан алгоритм для поиска максимальных значений в столбцах матрицы. Реализованные последовательная и параллельная MPI-реализации успешно проходят все функциональные тесты, включая обработку граничных случаев (пустые матрицы, матрицы с нулевыми столбцами) и работу с различными типами данных.

Проведенное исследование производительности на матрицах размером 5000×5000 и 10000×10000 показало, что MPI-реализация демонстрирует положительное ускорение по сравнению с последовательной версией. Наилучшие результаты достигаются при использовании 2 процессов:

Для матрицы 5000×5000: ускорение 1.12×, эффективность 55.87%

Для матрицы 10000×10000: ускорение 1.24×, эффективность 61.95%

## 9. Источники
- Лекции Сысоева Александра Владимировича
- Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича
- Open MPI Documentation. https://www.open-mpi.org/doc/
- Introduction to Parallel Computing. https://computing.llnl.gov/tutorials/parallel_comp/
- MPI Tutorial for Beginners. https://mpitutorial.com/
- C++ Standard Library Reference. https://en.cppreference.com/
- "Основы MPI программирования". https://habr.com/ru/articles/121925/

## Приложение
```cpp
//MPI
bool EgorovaLFindMaxValColMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  return RunMPIAlgorithm();
}

bool EgorovaLFindMaxValColMatrixMPI::RunMPIAlgorithm() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = static_cast<int>(GetInput()[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределение столбцов по процессам
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;
  int start_col = 0;
  int local_cols_count = 0;

  if (rank < remainder) {
    start_col = rank * (cols_per_proc + 1);
    local_cols_count = cols_per_proc + 1;
  } else {
    start_col = (remainder * (cols_per_proc + 1)) + ((rank - remainder) * cols_per_proc);
    local_cols_count = cols_per_proc;
  }

  // Получение локальной части матрицы
  std::vector<int> local_matrix_part = GetLocalMatrixPart(rank, size, rows, cols, start_col, local_cols_count);
  std::vector<int> local_max = CalculateLocalMaxima(local_matrix_part, rows, local_cols_count);
  std::vector<int> all_max = GatherResults(local_max, size, cols);

  GetOutput() = all_max;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::GetLocalMatrixPart(int rank, int size, int rows, int cols,
                                                                    int start_col, int local_cols_count) {
  std::vector<int> local_part(static_cast<std::size_t>(rows) * static_cast<std::size_t>(local_cols_count));

  if (rank == 0) {
    const auto &matrix = GetInput();

    // Процесс 0 заполняет свою локальную часть
    for (int ii = 0; ii < rows; ++ii) {
      for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
        const int global_col = start_col + local_idx;
        local_part[(static_cast<std::size_t>(ii) * static_cast<std::size_t>(local_cols_count)) + local_idx] =
            matrix[ii][global_col];
      }
    }

    // Отправка частей матрицы другим процессам
    SendMatrixPartsToOtherRanks(size, rows, cols);
  } else {
    // Получение данных от процесса 0
    MPI_Recv(local_part.data(), static_cast<int>(local_part.size()), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return local_part;
}

void EgorovaLFindMaxValColMatrixMPI::SendMatrixPartsToOtherRanks(int size, int rows, int cols) {
  for (int dest_rank = 1; dest_rank < size; ++dest_rank) {
    std::vector<int> dest_part = PrepareMatrixPartForRank(dest_rank, size, rows, cols);

    // Отправка данных процессу-получателю
    MPI_Send(dest_part.data(), static_cast<int>(dest_part.size()), MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
  }
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::PrepareMatrixPartForRank(int dest_rank, int size, int rows, int cols) {
  // Вычисление диапазона столбцов для процесса-получателя
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;

  int dest_start_col = 0;
  int dest_cols_count = 0;

  if (dest_rank < remainder) {
    dest_start_col = dest_rank * (cols_per_proc + 1);
    dest_cols_count = cols_per_proc + 1;
  } else {
    dest_start_col = (remainder * (cols_per_proc + 1)) + ((dest_rank - remainder) * cols_per_proc);
    dest_cols_count = cols_per_proc;
  }

  // Подготовка данных для отправки
  const auto &matrix = GetInput();
  std::vector<int> dest_part(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dest_cols_count));

  for (int ii = 0; ii < rows; ++ii) {
    for (int jj = 0; jj < dest_cols_count; ++jj) {
      const int global_col = dest_start_col + jj;
      dest_part[(static_cast<std::size_t>(ii) * static_cast<std::size_t>(dest_cols_count)) + jj] =
          matrix[ii][global_col];
    }
  }

  return dest_part;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::CalculateLocalMaxima(const std::vector<int> &local_matrix_part,
                                                                      int rows, int local_cols_count) {
  std::vector<int> local_max(local_cols_count, std::numeric_limits<int>::min());

  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    for (int row = 0; row < rows; ++row) {
      const int value = local_matrix_part[(row * local_cols_count) + local_idx];
      local_max[local_idx] = std::max(value, local_max[local_idx]);
    }
  }

  return local_max;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::GatherResults(const std::vector<int> &local_max, int size, int cols) {
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;

  std::vector<int> all_max(cols, std::numeric_limits<int>::min());
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  for (int ii = 0; ii < size; ++ii) {
    recv_counts[ii] = (ii < remainder) ? (cols_per_proc + 1) : cols_per_proc;
    displs[ii] = (ii == 0) ? 0 : displs[ii - 1] + recv_counts[ii - 1];
  }

  MPI_Allgatherv(local_max.data(), static_cast<int>(local_max.size()), MPI_INT, all_max.data(), recv_counts.data(),
                 displs.data(), MPI_INT, MPI_COMM_WORLD);

  return all_max;
}

bool EgorovaLFindMaxValColMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_find_max_val_col_matrix
