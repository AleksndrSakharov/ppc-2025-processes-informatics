# Сумма значений по столбцам матрицы
**Студент:** Зенин Антон Алексеевич.
**Группа:** 3823Б1ФИ1
**Технология:** SEQ | MPI
**Вариант:** 12 
---

## 1. Введение
Цель задачи состоит в том, чтобы вычислить сумму значений в каждом столбце матрицы, используя два подхода:

1. **Sequential (SEQ)** реализация
2. **Parallel MPI реализация** с использованием нескольких процессов

Целью работы является: 
- разработка и внедрение обеих версий,
- проверка корректности с помощью функциональных тестов,
- измерение и анализ производительности,
- оценка параллельной эффективности.

---

## 2. Постановка задачи

Дана матрица `A` размером `R × C`, представленная в виде вектора в порядке row-major. R - число строк, C - число столбцов. Нужно вычислить:

$$
S_j = \sum_{i=1}^{R} A_{i,j},\quad j = 1..C
$$

**Входные данные**

- `InType = tuple<size_t rows, size_t columns, vector<double> data>`  
- Размер данных = `rows * columns`

**Выходные данные** 

- `OutType = vector<double>` of size `columns`

Особенности: 

- Элементы имеют тип `double`
- Входной формат проверяется перед вычислением 

--- 

## 3. Описание алгоритма SEQ (Sequential)

Последовательный алгоритм выполняет простой двойной цикл по строкам и столбцам:

for row in rows:
    for col in columns:
        sum[col] += matrix[row * columns + col]

Временная сложность:

$$
O(R \cdot C)
$$

Использование памяти:

- Входная матрица: `R*C*sizeof(double)`
- Выходные данные: `C*sizeof(double)`

Для performance тестов размер матрицы был: **6000 x 6000 (36,000,000 элементов)**

---

## 4. Схема распараллеливания (MPI)

## Стратегия

Матрица распределяется по столбцам, чтобы каждый процесс обрабатывал свой набор столбцов.
Используется схема: 
- `base = C / world_size`
- `rest = C % world_size`
- `my_cols = base + (rank < rest ? 1 : 0)`

- На процессе 0 формируется отправной буфер с переставленными по процессам столбцами.    
- Используется `MPI_Scatterv` для передачи блоков.    
- Локальные вычисления: каждый процесс суммирует свои столбцы.    
- Сбор результатов обратно на процесс 0 - через `MPI_Gatherv`.    
- Финальная рассылка готового результата всем процессам через `MPI_Bcast`.  

---

## 5. Особенности реализации

## Структура кода

common/include/common.hpp — InType, OutType, формат входных и выходных данных  
seq/src/ops_seq.cpp — реализация последовательного алгоритма (SEQ)  
mpi/src/ops_mpi.cpp — реализация паралелльного алгоритма (MPI)  
tests/functional/main.cpp — functional tests  
tests/performance/main.cpp — performance tests  

## Классы

- `ZeninASumValuesByColumnsMatrixSEQ : BaseTask`
- `ZeninASumValuesByColumnsMatrixMPI : BaseTask`

## Соображения памяти

- SEQ:    
- Использует входную матрицу и выходной массив.  
- Память: `R * C * sizeof(double) + C * sizeof(double)`  

- MPI:
- Нулевой процесс дополнительно выделяет буфер sendbuf размером `R*C`.  
- Каждый процесс выделяет `local_block` размером `R * my_cols`.  
- Память не дублируется на всех процессах одновременно, матрица не рассылается полностью, только требуемые столбцы.  

---

## 6. Окружение

### Hardware
- CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz  
- Cores: 6  
- RAM: 32 GB  
- OS: Windows 10 Pro

### Toolchain
- Compiler: `C++20`   
- MPI: OpenMPI  
- Build type: Release

### Environment variables

PPC_NUM_THREADS = 1    
PPC_NUM_PROCS = 4 (для запуска с использованием MPI)

## Данные
- Используются 15 функциональных тестов со следующими размерами матриц: (3, 3), (2, 5), (10, 70), (1, 1), (1, 100), (100, 1), 
 (1000, 1000), (10, 2), (5, 3), (4, 5), (4, 3), (10000, 3), (3, 10000), (500, 1), (1, 500).  
- Тест на производительность использует матрицу размером **6000 строк × 6000 столбцов** 

---

## Результаты и выводы

### 7.1 Корректность
Корректность была протестирована с помощью функциональных тестов:

- 15 входных матриц
- Обе реализации SEQ и MPI были проверены
- Все тесты были пройдены как в однопроцессорном, так и в многопроцессорном режимах

### 7.2 Производительность

#### Результаты performance-тестов

| Mode (normal run) | Time (s) | Speedup vs SEQ | Efficiency (4 proc) |
|-------------------|----------|----------------|---------------------|
| **SEQ pipeline**  | 0.026123 |      1.0       |          —          |
| **SEQ task_run**  | 0.030003 |      1.0       |          —          |
| **MPI pipeline**  | 0.511198 |     0.051      |         1.28%        |
| **MPI task_run**  | 0.505920 |     0.059      |         1.48%        |

---

| Mode (mpiexec -n 4) | Time (s) | Speedup vs SEQ | Efficiency (4 proc) |
|---------------------|----------|----------------|---------------------|
| **SEQ pipeline**    | 0.047043 |       1.00     |          —          |
| **SEQ task_run**    | 0.055815 |       1.00     |          —          |
| **MPI pipeline**    | 0.465189 |      0.101     |        2.53%        |
| **MPI task_run**    | 0.487792 |      0.114     |        2.86%        |

---

## Анализ результатов

- Результаты показывают, что MPI-реализация уступает последовательной по производительности. 
- Операции суммирования - простые. MPI накладные расходы (Scatterv, Gatherv, Bcast) оказываются дороже самих вычислений. 
- Затраты на вычисления малы относительно времени коммуникаций.  
- MPI медленнее SEQ, поскольку еще дополнительно происходит: распределение данных, сбор результатов, подготовка буфера, синхронизация процессов.  

---

## 8. Выводы
- Реализованы две версии решения задачи: последовательная SEQ и параллельная MPI.  
- Все функциональные тесты успешно пройдены, корректность подтверждена.  
- Для матрицы 6000×6000 последовательная версия демонстрирует производительность выше, чем MPI.  
- MPI-версия показывает минимальную эффективность (1–3%), что объясняется: большими накладными расходами на коммуникации, 
  высокой стоимостью подготовки данных, низкой вычислительной сложностью задачи.  
---

## 9. Источники
- cppreference.com - https://en.cppreference.com
- Документация по OpenMPI - https://www.open-mpi.org/doc
- Лекции по параллельному программированию ННГУ им. Лобачевского
- Практические занятия по параллельному программированию ННГУ им. Лобачевского

---

## 10. Приложение. Код MPI реализации
```cpp

class ZeninASumValuesByColumnsMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZeninASumValuesByColumnsMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void FillSendBuffer(const std::vector<double> &mat, std::vector<double> &sendbuf, size_t rows, size_t cols,
                             size_t base, size_t rest, int world_size);
};

void ZeninASumValuesByColumnsMatrixMPI::FillSendBuffer(const std::vector<double> &mat, std::vector<double> &sendbuf,
                                                       size_t rows, size_t cols, size_t base, size_t rest,
                                                       int world_size) {
  size_t pos = 0;
  for (int proc = 0; proc < world_size; proc++) {
    auto proc_size = static_cast<size_t>(proc);
    size_t pc_begin = (proc_size * base) + (std::cmp_less(proc_size, rest) ? proc_size : rest);
    size_t pc_end = pc_begin + (base + (std::cmp_less(proc_size, rest) ? 1 : 0));
    for (size_t col = pc_begin; col < pc_end; col++) {
      for (size_t row = 0; row < rows; row++) {
        sendbuf[pos++] = mat[(row * cols) + col];
      }
    }
  }
}

bool ZeninASumValuesByColumnsMatrixMPI::RunImpl() {
  auto rows = static_cast<size_t>(std::get<0>(GetInput()));
  auto cols = static_cast<size_t>(std::get<1>(GetInput()));
  const std::vector<double> &mat = std::get<2>(GetInput());

  std::vector<double> &global_sum = GetOutput();

  int rank = 0;
  int world_size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const size_t base = cols / static_cast<size_t>(world_size);
  const size_t rest = cols % static_cast<size_t>(world_size);

  const size_t my_cols = base + (std::cmp_less(static_cast<size_t>(rank), rest) ? 1 : 0);

  std::vector<int> sendcounts(static_cast<size_t>(world_size));
  std::vector<int> displs(static_cast<size_t>(world_size));
  if (rank == 0) {
    int offset = 0;
    for (int proc = 0; proc < world_size; proc++) {
      size_t pc = base + (std::cmp_less(static_cast<size_t>(proc), rest) ? 1 : 0);
      sendcounts[proc] = static_cast<int>(pc * rows);
      displs[proc] = offset;
      offset += sendcounts[proc];
    }
  }

  std::vector<double> sendbuf;
  if (rank == 0) {
    sendbuf.resize(rows * cols);
    FillSendBuffer(mat, sendbuf, rows, cols, base, rest, world_size);
  }
  std::vector<double> local_block(rows * my_cols);
  MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_block.data(),
               static_cast<int>(local_block.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_sum(my_cols, 0.0);
  for (size_t col_id = 0; col_id < my_cols; col_id++) {
    for (size_t row_id = 0; row_id < rows; row_id++) {
      local_sum[col_id] += local_block[(col_id * rows) + row_id];
    }
  }
  std::vector<int> recvcounts(static_cast<size_t>(world_size));
  std::vector<int> recvdispls(static_cast<size_t>(world_size));

  if (rows == 0) {
    throw std::runtime_error("Matrix has zero rows");
  }

  if (rank == 0) {
    size_t offset = 0;
    for (int proc = 0; proc < world_size; proc++) {
      recvcounts[proc] = sendcounts[proc] / static_cast<int>(rows);
      recvdispls[proc] = static_cast<int>(offset);
      offset += static_cast<size_t>(recvcounts[proc]);
    }
    global_sum.assign(cols, 0.0);
  }

  MPI_Gatherv(local_sum.data(), static_cast<int>(my_cols), MPI_DOUBLE, global_sum.data(), recvcounts.data(),
              recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  global_sum.resize(cols);
  MPI_Bcast(global_sum.data(), static_cast<int>(cols), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}