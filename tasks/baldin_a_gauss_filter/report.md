# Линейная фильтрация изображений (горизонтальное разбиение). Ядро Гаусса 3x3.
- Студент: Балдин Андрей Леонидович, 3823Б1ФИ3
- Технология: SEQ, MPI 
- Вариант: 26

---

## 1. Введение
Фильтрация изображений -  одна из самых частых операций в компьютерном зрении и обработке сигналов. Размытие по Гауссу используется для уменьшения шума и детализации изображения.
Поскольку операция свертки является локальной (значение пикселя зависит только от его соседей), задача хорошо поддается распараллеливанию.
**Цель работы** — реализовать параллельный алгоритм фильтрации изображения с использованием MPI, применив стратегию декомпозиции данных по строкам, и сравнить его производительность с последовательной версией.

---

## 2. Постановка задачи
**Задано**: 
- Входное изображение, представленное как структура `ImageData` (ширина `W`, высота `H`, количество каналов `C`, массив пикселей `std::vector<uint8_t>`)

**Требуется**:
Применить фильтр Гаусса к изображению. Для каждого пикселя `(x, y)` новое значение вычисляется как взвешенная сумма соседей.
Формула ядра (с нормировочным коэффициентом 16):
```math
K = \frac{1}{16}
\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}
```

**Выходные данные**:
- Изображение тех же размеров (`W`, `H`) и с тем же количеством каналов (`C`), что и входное.
- Тип данных пикселей сохраняется (`uint8_t`).

**Ограничения и условия**:
- Обработка граничных пикселей: использовать дублирование крайнего пикселя.
- Тип данных пикселя: `uint8_t` (0-255).
- Параллелизация через горизонтальное разбиение.

---

## 3. Базовый алгоритм (последовательный)

Последовательная версия выполняет классическую свертку.
1.  Создается копия изображения для записи результата.
2.  Проходим по всем пикселям изображения (циклы по `H`, `W`, `C`).
3.  Для каждого пикселя рассматриваем окрестность `3 x 3`.
4.  Вычисляем сумму произведений значений пикселей окрестности на соответствующие коэффициенты ядра.
5.  Если сосед выходит за границы изображения, берется значение ближайшего граничного пикселя (`std::clamp`).
6.  Полученную сумму делим на 16 и записываем в результирующее изображение.

---

## 4.  Схема распараллеливания

Для распараллеливания используется **декомпозиция данных по строкам**. Изображение разрезается на горизонтальные полосы, и каждый процесс обрабатывает свою часть.

### 4.1. Оптимизация:
Ядро Гаусса `3 x 3` можно представить как произведение двух одномерных векторов:
```math
$$
\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \times \begin{bmatrix} 1 & 2 & 1 \end{bmatrix}
$$
```
Это позволяет заменить одну 2D-свертку (9 умножений на пиксель) на две 1D-свертки (3 + 3 = 6 умножений на пиксель).

1.  **Горизонтальный проход:** Свертка по строкам вектором $[1, 2, 1]$.
2.  **Вертикальный проход:** Свертка результата по столбцам вектором $[1, 2, 1]^T$.

### 4.2. Распределение данных
Для обработки строк на границе полосы процессу требуются данные из соседних полос.
Вместо явного обмена границами (`Send/Recv`) после распределения, реализована стратегия распределения с перекрытием:
- Корневой процесс при вычислении смещений (`displs`) и размеров (`counts`) для `MPI_Scatterv` добавляет каждому процессу по одной дополнительной строке сверху и снизу (если эти строки существуют).
- Таким образом, каждый процесс сразу получает все необходимые данные для обработки своей части изображения.

### 4.3. Сборка результатов
После вычислений процессы имеют результат только для своих частей. Корневой процесс собирает итоговое изображение с помощью `MPI_Gatherv`, конкатенируя обработанные полосы.

---

## 5. Детали реализации

**Ключевые файлы:**
- `baldin_a_gauss_filter/seq/include/ops_seq.hpp` и `baldin_a_gauss_filter/seq/src/ops_seq.cpp` - последовательная реализация.
- `baldin_a_gauss_filter/mpi/include/ops_mpi.hpp` и `baldin_a_gauss_filter/mpi/src/ops_mpi.cpp` - параллельная реализация.
- `baldin_a_gauss_filter/common/include/common.hpp` - общие определения типов и интерфейсов.

**Основные функции:**
- `CalculatePartitions` - вспомогательная функция для расчета распределения нагрузки. Вычисляет, какие строки (включая перекрытия) отправить каждому процессу.
- `ComputeHorizontalPass` - выполняет одномерную свертку по горизонтали. Результат сохраняется в промежуточный буфер типа `uint16_t`, чтобы избежать переполнения перед финальным делением.
- `ComputeVerticalPass` - выполняет одномерную свертку по вертикали, используя данные из промежуточного буфера. Результат нормируется и приводится к `uint8_t`.

**Особенности:**
1.  Использование `uint16_t` для промежуточных вычислений важно, так как сумма взвешенных пикселей (макс `255 * 4 = 1020`) не помещается в `uint8_t`.
2. **Граничные условия**:
    - Внутри локального буфера процесса используется `std::clamp` для обработки левой и правой границ картинки.
    - Верхняя и нижняя границы обрабатываются корректно благодаря передаче перекрывающихся строк. Если процесс обрабатывает самый верх или низ изображения, алгоритм дублирует соответствующие строки (логика внутри `ComputeVerticalPass`).
3.  **Тестирование**:
    - Для генерации детерминированных, но "случайных" изображений в тестах используется `std::mt19937` с сидом, зависящим от параметров изображения (`width + height + channels`). Это гарантирует, что все MPI-процессы работают с идентичными входными данными без необходимости пересылать их.
4. **Валидация входных данных (`ValidationImpl`):**
	- Размеры изображения (ширина, высота) и количество каналов должны быть строго больше нуля.
	- Фактический размер вектора пикселей (`pixels.size()`) должен точно соответствовать произведению измерений: `W * H * C`. Это гарантирует отсутствие выхода за границы массива при доступе по индексу.
5.  **Обработка результатов (`PostProcessingImpl`):**
    Согласно логике параллельного алгоритма, сборка итогового изображения (`MPI_Gatherv`) происходит в методе `RunImpl` только на корневом процессе. Однако архитектура тестового фреймворка требует наличия результата на всех процессах для верификации корректности.
    -   В метод `PostProcessingImpl` вынесена логика синхронизации результатов.
    -   Корневой процесс сначала рассылает размеры через `MPI_Bcast`.
    -   Остальные процессы выделяют память под результат.
    -   Затем происходит рассылка массива пикселей всем процессам.
    Это позволяет корректно завершить выполнение тестов на всех рангах, не ухудшая показатели производительности самого алгоритма фильтрации.

---

## 6. Экспериментальная установка
- **Аппаратное обеспечение**: 
	Apple M1 Pro (8 ядер CPU: 6 производительных + 2 эффективных, 10 ядер GPU)  
	ОЗУ — 16 ГБ LPDDR5
- **Операционная система:** macOS Sonoma 14.5
- **Компилятор:** Apple Clang 21.1.4 (Homebrew)
- **MPI-библиотека:** Open MPI 5.0.8 (Homebrew)
- **Тип сборки:** Release (`-O3`)

---

## 7.  Результаты и обсуждение
### 7.1 Корректность
Корректность проверена на наборе функциональных тестов, включающем:
- Изображения малого размера (`1 x 1`, `3 x 3`).
- "Полоски" (`100 x 1`,  `1 x 100`) для проверки обработки границ.
- Нестандартные размеры (`127 x 113`).
- Различное количество каналов (`1`, `3`, `4`).
Результат MPI версии побайтово совпадает с результатом последовательной версии.

### 7.2 Производительность
Для тестирования производительности использовалась генерация изображения размером `3000 x 3000` пикселей (3 цветовых канала).

**Результаты замеров:**

| Mode | Count | Time, s | Speedup | Efficiency |
| ---- | ----- | ------- | ------- | ---------- |
| seq  | 1     | 0.0686  | 1.00    | N/A        |
| mpi  | 2     | 0.0408  | 1.681   | 84.05%     |
| mpi  | 3     | 0.0328  | 2.091   | 69.70%     |
| mpi  | 4     | 0.0284  | 2.415   | 48.30%     |
| mpi  | 8     | 0.0372  | 1.844   | 23.05%     |

**Анализ результатов:**
1. Наблюдается стабильный рост ускорения при увеличении числа процессов от 1 до 4. Минимальное время выполнения достигнуто на 4 процессах, что дает ускорение в 2.42 раза.
2. Эффективность падает с ростом числа процессов. Это объясняется тем, что при фиксированном размере задачи доля полезных вычислений на один процесс уменьшается, а накладные расходы на инициализацию и коммуникацию остаются или растут.
3. При запуске на 8 процессах время выполнения увеличилось по сравнению с 4 процессами. Это связано с несколькими факторами:
    * Время, затрачиваемое на передачу данных (`Scatterv`, `Gatherv`), начинает превышать время самих вычислений.
    * Создание и синхронизация 8 процессов MPI занимают ощутимое время по сравнению с общим временем работы задачи.

---

## 8. Заключение

Реализован параллельный фильтр Гаусса с использованием MPI.
- Применена декомпозиция по строкам с перекрытием, что позволило избежать обменов сообщениями в процессе вычислений.
- Использована оптимизация сепарабельности ядра, снижающая алгоритмическую сложность.
- Достигнуто ускорение в 2.4 раза на 4 процессах, что подтверждает эффективность подхода для задач обработки изображений.

---

## 9. Источники
- Лекции и практики курса "Параллельное программирование"
- Алгоритм фильтра Гаусса: [https://ru.wikipedia.org/wiki/Размытие_по_Гауссу](https://ru.wikipedia.org/wiki/Размытие_по_Гауссу)

---

## 10. Приложение

```cpp
namespace {

void CalculatePartitions(int size, int height, int width, int channels, std::vector<int> &counts,
                         std::vector<int> &displs, std::vector<int> &real_counts) {
  const int rows_per_proc = height / size;
  const int remainder = height % size;

  int current_global_row = 0;
  const int row_size_bytes = width * channels;

  for (int i = 0; i < size; i++) {
    int rows = rows_per_proc + ((i < remainder) ? 1 : 0);
    real_counts[i] = rows;

    int send_start_row = std::max(0, current_global_row - 1);
    int send_end_row = std::min(height - 1, current_global_row + rows);

    int rows_to_send = send_end_row - send_start_row + 1;

    counts[i] = rows_to_send * row_size_bytes;
    displs[i] = send_start_row * row_size_bytes;

    current_global_row += rows;
  }
}

void ComputeHorizontalPass(int rows, int width, int channels, const std::vector<uint8_t> &src,
                           std::vector<uint16_t> &dst) {
  constexpr std::array<int, 3> kKernel = {1, 2, 1};

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < width; col++) {
      for (int ch = 0; ch < channels; ch++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          int n_col = std::clamp(col + k, 0, width - 1);
          sum += src[(((row * width) + n_col) * channels) + ch] * kKernel.at(k + 1);
        }
        dst[(((row * width) + col) * channels) + ch] = static_cast<uint16_t>(sum);
      }
    }
  }
}

void ComputeVerticalPass(int real_rows, int recv_rows, int width, int channels, int row_offset,
                         const std::vector<uint16_t> &src, std::vector<uint8_t> &dst) {
  constexpr std::array<int, 3> kKernel = {1, 2, 1};

  for (int i = 0; i < real_rows; i++) {
    int local_row = row_offset + i;

    for (int col = 0; col < width; col++) {
      for (int ch = 0; ch < channels; ch++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          int neighbor_row = local_row + k;
          neighbor_row = std::clamp(neighbor_row, 0, recv_rows - 1);

          sum += src[(((neighbor_row * width) + col) * channels) + ch] * kKernel.at(k + 1);
        }
        dst[(((i * width) + col) * channels) + ch] = static_cast<uint8_t>(sum / 16);
      }
    }
  }
}

}  // namespace

bool BaldinAGaussFilterMPI::RunImpl() {
  ImageData &input = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;

  if (rank == 0) {
    width = input.width;
    height = input.height;
    channels = input.channels;
  }
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  std::vector<int> real_counts(size);

  CalculatePartitions(size, height, width, channels, counts, displs, real_counts);

  int my_real_rows = real_counts[rank];
  int my_recv_rows = counts[rank] / (width * channels);

  size_t row_size_bytes = static_cast<size_t>(width) * channels;
  std::vector<uint8_t> local_buffer(static_cast<size_t>(my_recv_rows) * row_size_bytes);
  std::vector<uint16_t> horiz_buffer(static_cast<size_t>(my_recv_rows) * row_size_bytes);
  std::vector<uint8_t> result_buffer(static_cast<size_t>(my_real_rows) * row_size_bytes);

  MPI_Scatterv(rank == 0 ? input.pixels.data() : nullptr, counts.data(), displs.data(), MPI_UINT8_T,
               local_buffer.data(), counts[rank], MPI_UINT8_T, 0, MPI_COMM_WORLD);

  ComputeHorizontalPass(my_recv_rows, width, channels, local_buffer, horiz_buffer);

  int row_offset = (rank == 0) ? 0 : 1;
  ComputeVerticalPass(my_real_rows, my_recv_rows, width, channels, row_offset, horiz_buffer, result_buffer);

  if (rank == 0) {
    GetOutput().width = width;
    GetOutput().height = height;
    GetOutput().channels = channels;
    GetOutput().pixels.resize(static_cast<size_t>(width) * height * channels);
  }

  std::vector<int> recv_counts(size);
  std::vector<int> recv_displs(size);

  if (rank == 0) {
    int current_disp = 0;
    int row_bytes_int = width * channels;
    for (int i = 0; i < size; i++) {
      recv_counts[i] = real_counts[i] * row_bytes_int;
      recv_displs[i] = current_disp;
      current_disp += recv_counts[i];
    }
  }

  MPI_Gatherv(result_buffer.data(), static_cast<int>(result_buffer.size()), MPI_UINT8_T,
              (rank == 0 ? GetOutput().pixels.data() : nullptr), recv_counts.data(), recv_displs.data(), MPI_UINT8_T, 0,
              MPI_COMM_WORLD);

  return true;
}
```