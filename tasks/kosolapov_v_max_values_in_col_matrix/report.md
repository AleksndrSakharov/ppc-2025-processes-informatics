# Отчёт по лабораторной работе  
## «Параллельное нахождение максимальных значений по столбцам матрицы с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Косолапов Виталий Андреевич 3823Б1ФИ1
**Вариант:** 16
---

## Введение

Задача нахождения максимальных значений по столбцам матрицы является фундаментальной операцией в вычислительной математике и анализе данных. Эта операция широко используется в различных приложениях, включая обработку сигналов, машинное обучение, научные вычисления и бизнес-аналитику. В контексте обработки больших данных, когда матрицы могут содержать миллионы элементов, последовательный алгоритм становится недостаточно эффективным. Одним из ключевых методов ускорения вычислений является распараллеливание. В данной работе рассматривается реализация алгоритма нахождения максимальных значений по столбцам матрицы, как в последовательной, так и в параллельной формах, с использованием технологии MPI (Message Passing Interface).  

Цель работы — продемонстрировать принципы параллельного программирования, а также сравнить эффективность и корректность работы MPI-версии с последовательной реализацией.

---

## Постановка задачи

Необходимо реализовать нахождение **максимального значения по всем столбцам матрицы**.  

Дана матрица A размером M × N найти вектор максимумов V размером N, где каждый элемент V[i] представляет максимальное значение в i-м столбце матрицы A.

$$
V[i] = max{A[0][i], A[1][i], ..., A[M-1][i]} для всех i ∈ [0, N-1]
$$

### Требуется:
1. Реализовать **последовательную версию** вычисления.  
2. Реализовать **параллельную версию** с использованием **MPI**.  
3. Провести сравнение времени выполнения и подтвердить корректность вычислений.  

---

## Описание алгоритма

Алгоритм поиска максимального значения по столбцам матрицы можно описать следующим образом:

1. Проверка корректности входных данных: все строки матрицы должны быть одной длины.  
2. Инициализация результата пустым вектором длины N.  
3. Для всех элементов \( i \) от 0 до \( N-1 \): 
   - Установить значение `temp_max` = A[0][i];
   - Для всех элементов \( j \) от 0 до \( M-1 \):
     - Если A[j][i] > `temp_max`:
       - Установить значение `temp_max` = A[j][i];  
   - Установить i-й элемент результирующего вектора равным `temp_max`;
4. Вернуть полученное значение как результат поиска максимума.

---

## Описание схемы параллельного алгоритма

В параллельной версии с использованием **MPI**:

1. **Инициализация MPI** — каждый процесс получает свой ранг `rank` и общее число процессов `processes_count`.  
2. **Разделение данных:**  
   - Строки равномерно распределяются между процессами.  
   - Если количество строк не делится нацело, лишние элементы распределяются по первым процессам.  
3. **Вычисление частичных результатов:**  
   Каждый процесс вычисляет максимальное значение своей части строк по столбцам.  
4. **Сбор результатов:**  
   Используется `MPI_Allreduce`, который берёт локальные результаты(вектора с максимумом по столбцам) и формирует из них глобальный максимум(итоговый результат).
5. **Вывод результата и завершение работы MPI.**

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++17** с использованием **библиотеки MPI**.  
Класс `KosolapovVMaxValuesInColMatrixMPI` реализует интерфейс параллельного поиска максимальных значений.  

Основные этапы:
- Инициализация и валидация входных данных;  
- Определение диапазона элементов, обрабатываемых каждым процессом;  
- Передача строк с которыми работает каждый процесс при помощи `MPI_Send` и `MPI_Recv`
- Локальное вычисление максимумов;  
- Объединение частичных результатов между всеми процессами при помощи `MPI_Allreduce`;  
- Возврат итогового значения.

### Ключевые функции:
- `MPI_Comm_rank`, `MPI_Comm_size` — определяют номер процесса и общее количество процессов;  
- `MPI_Send` — отправляет сообщение(данные) другому процессу;
- `MPI_Recv` — принимает сообщение(данные) от другого процесса;
- `MPI_Allreduce` — коллективная операция, которая выполняет редукцию (суммирование, максимум и т.д.) над данными всех процессов и рассылает результат всем процессам;
- `MPI_Bcast` - рассылает данные от одного процесса всем остальным процессам.

### Валидация данных
Не допускаются матрицы с разными длинами строк.

---

## Результаты экспериментов

### Условия экспериментов
- Размеры матриц: \(\(10^5\)*\(10^5\)\), \(\(10^4\)*\(10^4\)\), \(\(2*\(10^4\)\)*\(2*\(10^4\)\)\) элементов.  
- Среда выполнения: Windows, MPI (4 процесса).  
- Измерение времени проводилось встроенными средствами тестового фреймворка GoogleTest.  

---

### Результаты при размере матрицы \(\(10^4\)*\(10^4\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.00569     | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.00593     | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.00441     | 1.29      | 32.25%        |
| MPI task_run    | 4                | 0.00374     | 1.59      | 39.75%        |

---

### Результаты при размере матрицы \(\(2*\(10^4\)\)*\(2*\(10^4\)\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.0283      | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.0272      | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.0203      | 1.39      | 34.75%        |
| MPI task_run    | 4                | 0.0199      | 1.37      | 34.25%        |

---

### Результаты при размере матрицы \(\(10^5\)*\(10^5\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.902       | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.900       | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.437       | 2.06      | 51.5%         |
| MPI task_run    | 4                | 0.439       | 2.05      | 51.25%        |
| MPI pipeline    | 6                | 0.376       | 2.39      | 39.83%        |
| MPI task_run    | 6                | 0.380       | 2.36      | 39.33%        |

---

## Подтверждение корректности
Результаты последовательной и параллельной версий совпадают для всех протестированных вариаций матриц, включая граничный случаи(матрица 1*1, матрица с одной строчкой, матрица с одним столбцом). Ошибок не наблюдалось, при тестировании возвращался ожидаемый результат. Таким образом, реализация корректна.

---

## Выводы
1. Реализованы две версии алгоритма — последовательная и MPI.  
2. Параллельная реализация успешно масштабируется и даёт ускорение при увеличении размера данных.  
3. При малых объёмах данных накладные расходы MPI превышают выигрыш от распараллеливания.  
4. Алгоритм корректен и демонстрирует ожидаемое поведение при всех размерах входных данных.  
5. Эффективность увеличивается, при увелечение входных данных, но снижается при увелечение количества процессов, что связано с увелечение накладных расходов.

---

## Заключение
В ходе лабораторной работы был реализован и протестирован алгоритм поиска максимального элемента по столбцам матрицы с использованием технологии MPI. Проведён сравнительный анализ с последовательной версией, подтверждена корректность вычислений, проведены измерения ускорения и эффективности. Полученные результаты демонстрируют эффективность использования параллельных вычислений для ресурсоёмких задач.

---

## Список литературы

1. Параллельные вычисления. MPI Tutorial. — https://mpitutorial.com  
2. Документация по OpenMPI — https://www.open-mpi.org/doc/ 
3. Лекции Сысоева Александра Владимировича
4. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича

---

## Приложение. Исходный код

### Последовательная версия

```cpp

#include "kosolapov_v_max_values_in_col_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixSEQ::KosolapovVMaxValuesInColMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()){
    return false;
  }
  for (size_t i = 0; i < matrix.size() - 1; i++) {
    if (matrix[i].size() != matrix[i + 1].size()) {
      return false;
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput()[0].size());
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  for (size_t i = 0; i < matrix[0].size(); i++) {
    int temp_max = matrix[0][i];
    for (const auto &row : matrix) {
      temp_max = std::max(row[i], temp_max);
    }
    GetOutput()[i] = temp_max;
  }
  return true;
}

bool KosolapovVMaxValuesInColMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kosolapov_v_max_values_in_col_matrix

```
---
### MPI-версия
```cpp
#include "kosolapov_v_max_values_in_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixMPI::KosolapovVMaxValuesInColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &matrix = GetInput();
    if (matrix.empty()) {
      return false;
    }
    for (size_t i = 0; i < matrix.size() - 1; i++) {
      if ((matrix[i].size() != matrix[i + 1].size()) || (matrix[i].empty())) {
        return false;
      }
    }
  }  
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &matrix = GetInput();
    if (!matrix.empty() && !matrix[0].empty()) {
      GetOutput().resize(matrix[0].size());
    }
  }  
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::RunImpl() {
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::vector<int>> local_matrix;
  int rows, columns;
  if (rank == 0) {
    const auto &matrix = GetInput();
    rows = static_cast<int>(matrix.size());
    columns = static_cast<int>(matrix[0].size());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int rows_per_proc = rows / processes_count;
  const int remainder = rows % processes_count;
  const int start = (rank * rows_per_proc) + std::min(rank, remainder);
  const int end = start + rows_per_proc + (rank < remainder ? 1 : 0);
  const int local_rows = end - start;

  if (rank == 0) {
    const auto &matrix = GetInput();
    local_matrix.resize(local_rows, std::vector<int>(columns));
    for (int i = 0; i < local_rows; i++) {
      local_matrix[i] = matrix[start + i];
    }

    for (int proc = 1; proc < processes_count; proc++) {
      const int proc_start = (proc * rows_per_proc) + std::min(proc, remainder);
      const int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
      const int proc_rows_count = proc_end - proc_start;
      for (int i = 0; i < proc_rows_count; i++) {
        MPI_Send(matrix[proc_start + i].data(), columns, MPI_INT, proc, i, MPI_COMM_WORLD);
      }
    }
  } else {
    local_matrix.resize(local_rows, std::vector<int>(columns));
    for (int i = 0; i < local_rows; i++) {
      MPI_Recv(local_matrix[i].data(), columns, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  auto local_maxs = CalculateLocalMax(local_matrix, columns);
  std::vector<int> global_maxs(columns);
  MPI_Allreduce(local_maxs.data(), global_maxs.data(), columns, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_maxs;
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::PostProcessingImpl() {
  return true;
}

std::vector<int> KosolapovVMaxValuesInColMatrixMPI::CalculateLocalMax(const std::vector<std::vector<int>> &matrix, const int columns) {
  std::vector<int> local_maxs(columns, std::numeric_limits<int>::min());
  for (const auto &row : matrix) {
    for (int i = 0; i < columns; i++) {
      if (row[i] > local_maxs[i]) {
        local_maxs[i] = row[i];
      }
    }
  }
  return local_maxs;
}
}  // namespace kosolapov_v_max_values_in_col_matrix

```