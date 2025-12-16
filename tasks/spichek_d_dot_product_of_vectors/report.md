# Отчёт по лабораторной работе  
## «Параллельное вычисление скалярного произведения векторов с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Спичек Денис Игоревич 3823Б1ФИ1  
**Вариант:** 9  

---

## Введение

Современные вычислительные задачи часто требуют обработки больших объёмов данных за ограниченное время. Одним из ключевых методов ускорения вычислений является распараллеливание. В данной работе рассматривается реализация алгоритма вычисления скалярного произведения двух векторов как в последовательной, так и в параллельной формах с использованием технологии MPI (Message Passing Interface).  

Цель работы — продемонстрировать принципы параллельного программирования, а также сравнить эффективность и корректность работы MPI-версии с последовательной реализацией.

---

## Постановка задачи

Необходимо реализовать вычисление **скалярного произведения двух векторов одинаковой длины**.  

Скалярное произведение двух векторов $\mathbf{a} = (a_1, a_2, \dots, a_n)$ и $\mathbf{b} = (b_1, b_2, \dots, b_n)$ вычисляется по формуле:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

### Требуется:
1. Реализовать **последовательную версию** вычисления.  
2. Реализовать **параллельную версию** с использованием **MPI**.  
3. Провести сравнение времени выполнения и подтвердить корректность вычислений.  

---

## Описание алгоритма

Алгоритм вычисления скалярного произведения можно описать следующим образом:

1. Проверка корректности входных данных: векторы должны быть одной длины.  
2. Инициализация результата `dot_product = 0`.  
3. Для всех элементов $i$ от 0 до $n-1$:  
   - Вычислить частичный результат $a[i] \cdot b[i]$;  
   - Добавить его к сумме `dot_product`.  
4. Вернуть полученное значение как результат скалярного произведения.

---

## Описание схемы параллельного алгоритма

В параллельной версии с использованием **MPI**:

1. **Инициализация MPI** — каждый процесс получает свой ранг `rank` и общее число процессов `size`.  
2. **Разделение данных:** - Используется `MPI_Scatterv` для распределения векторов между процессами.
   - Если размер векторов не делится нацело на количество процессов, остаток равномерно распределяется между первыми процессами (балансировка нагрузки).  
3. **Вычисление частичных результатов:** Каждый процесс вычисляет скалярное произведение своей части векторов. Для суммирования используется тип `int64_t` во избежание переполнения.
4. **Сбор результатов:** Используется операция `MPI_Allreduce` с операцией `MPI_SUM` для суммирования частичных произведений всех процессов и получения итогового значения на каждом узле.  
5. **Вывод результата и завершение работы MPI.**

Схематично процесс можно изобразить так:

```
[Rank 0] --(Scatterv)--> [Rank 1..N]
   |                        |
(Calc)                   (Calc)
   \                        /
    -------> MPI_Allreduce (Sum) --> [Общий результат]
```

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++17** с использованием **библиотеки MPI**.  
Класс `SpichekDDotProductOfVectorsMPI` реализует интерфейс параллельного вычисления.  

**Ключевые особенности реализации:**
- Используется `MPI_Scatterv` для передачи данных только нужным процессам, что оптимизирует работу с памятью.
- Предварительный расчет смещений (`displs`) и размеров (`counts`) гарантирует, что все элементы вектора будут обработаны ровно один раз.
- Код оптимизирован для прохождения строгих проверок статического анализатора `clang-tidy` (использование `int64_t`, явные приведения типов, безопасные заголовки).

---

## Результаты экспериментов

### Условия экспериментов
- **Размер векторов:** $100,000,000$ ($10^8$) элементов.  
- **Среда выполнения:** Windows, MS-MPI.  
- **Оборудование:** Локальная рабочая станция.
- **Методика:** Измерение времени проводилось встроенными средствами тестового фреймворка (GoogleTest). Приведены усредненные значения времени выполнения (pipeline).

### Результаты замеров времени

| Кол-во процессов (MPI) | Время MPI (сек) | Время SEQ (сек) | Ускорение (отн. MPI-1) | Примечание |
|:---:|---:|---:|---:|:---|
| **1** | 0.2649 | 0.0276 | 1.00x | Базовое время MPI (накладные расходы) |
| **2** | 0.1917 | 0.0349 | 1.38x | Заметное снижение времени |
| **4** | 0.1958 | 0.0556 | 1.35x | Стабильный результат |
| **8** | 0.1608 | 0.0399 | 1.65x | Лучшее время выполнения |

### Анализ результатов

1. **Сравнение MPI vs SEQ:**
   Последовательная версия (`SEQ`) работает быстрее параллельной (`0.027с` против `0.16с`). Это ожидаемое поведение для простых операций (умножение + сложение) на одной машине с общей памятью. Накладные расходы на создание процессов MPI, выделение буферов и пересылку данных (`MPI_Scatterv`) превышают выигрыш от параллельного счета.

2. **Масштабируемость MPI:**
   Внутри самой MPI-реализации наблюдается **положительная динамика**. При увеличении числа процессов с 1 до 8 время выполнения сократилось с `0.2649 с` до `0.1608 с`, что дает ускорение примерно в **1.65 раза**. Это подтверждает, что алгоритм распараллелен корректно и при увеличении сложности вычислений (или объема данных) эффективность будет расти.

---

## Подтверждение корректности
Результаты тестов GoogleTest показывают статус `[ PASSED ]` для всех режимов запуска (1, 2, 4, 8 процессов). Итоговые суммы, вычисленные параллельно, полностью совпадают с результатами последовательного алгоритма.

---

## Выводы
1. Реализованы две версии алгоритма — последовательная и MPI.  
2. Параллельная реализация успешно масштабируется: время выполнения уменьшается при добавлении процессов (с 0.26с до 0.16с).  
3. Алгоритм корректно обрабатывает распределение нагрузки (остатки от деления размера вектора).
4. Продемонстрирована работа коллективных операций MPI (`Scatterv`, `Allreduce`).

---

## Приложение. Исходный код

### Последовательная версия (`ops_seq.cpp`)

```cpp
#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include <vector>

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsSEQ::SpichekDDotProductOfVectorsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsSEQ::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return vector1.size() == vector2.size();
}

bool SpichekDDotProductOfVectorsSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::RunImpl() {
  const auto &[vector1, vector2] = GetInput();
  
  if (vector1.empty()) {
    GetOutput() = 0;
    return true;
  }

  int dot_product = 0;
  for (size_t i = 0; i < vector1.size(); ++i) {
    dot_product += vector1[i] * vector2[i];
  }
  GetOutput() = dot_product;
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
```

### MPI-версия (`ops_mpi.cpp`)

```cpp
#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>
#include <cstdint>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsMPI::SpichekDDotProductOfVectorsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsMPI::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return vector1.size() == vector2.size();
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[v1, v2] = GetInput();
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(v1.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0 || (rank == 0 && static_cast<int>(v2.size()) != n)) {
    GetOutput() = 0;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  
  int base = n / size;
  int rem = n % size;

  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i * base) + std::min(i, rem);
  }

  int local_count = counts[rank];
  std::vector<int> lv1(local_count);
  std::vector<int> lv2(local_count);

  MPI_Scatterv(rank == 0 ? v1.data() : nullptr, counts.data(), displs.data(), MPI_INT, lv1.data(), local_count, MPI_INT,
               0, MPI_COMM_WORLD);
  MPI_Scatterv(rank == 0 ? v2.data() : nullptr, counts.data(), displs.data(), MPI_INT, lv2.data(), local_count, MPI_INT,
               0, MPI_COMM_WORLD);

  int64_t local_dot = 0;
  for (int i = 0; i < local_count; ++i) {
    local_dot += static_cast<int64_t>(lv1[i]) * lv2[i];
  }

  int64_t global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot);
  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
```