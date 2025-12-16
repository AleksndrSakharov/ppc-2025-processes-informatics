# Отчёт 

 - Студент: Шкребко Михаил Сергеевич, 3823Б1ФИ1
 - Технология: <SEQ | MPI>
 - Вариант: 23

# 1. Введение
Нужно подсчитать частоту символа в строке
Цель заключатеся в реализации последовательной и параллельной версии, а так же в их сравнении производительности.

# 2. Постановка задачи
Дан текст (строка) и символ для поиска. Требуется найти количество вхождений этого символа в текст.

Вход: text - строка произвольной длины, target_char - строка длины 1
Выход: count - целое число, количество вхождений символа в строку
Ограничения: строка не должна быть пустой, символ должен быть валидным

# 3. Описание алгоритма(Sequential)
Используем функцию count для подсчёта частоты символа в строке.
Функция возвращает количество вхождений заданного символа.

# 4. Описание алгоритма (MPI)
Распределение данных:
Исходная строка делится на части примерно одинакового размера.

Шаблон коммуникаций:
MPI_Bcast для передачи длины строки и искомого символа.
MPI_Scatterv для отправки каждому процессу его части строки.
MPI_Allreduce для суммирования подсчитанных частот и рассылки результата всем процессам.

Роли процессов:
Все процессы равны, за исключением процесса с рангом 0, который инициирует рассылку и распределение данных.

# 5. Результаты экспериментов и выводы
Было реализовано 4 функциональных теста:

Тест 1:   
Вход: ("Alolo polo", "l") → Выход: 3 

Тест 2: 
Вход: ("aramopma", "m") → Выход: 2

Тест 3: 
Вход: ("banana", "a") → Выход: 3

Тест 4: 
Вход: ("abcde", "z") → Выход: 0

Все тесты успешно проходят для реализаций SEQ и MPI, что подтверждает корректность алгоритма.

# 6. Производительность

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         |   1   |  0.145  |  1.00   |    N/A     |
| mpi         |   2   |  0.101  |  1.44   |   71.8%    |
| mpi         |   4   |  0.088  |  1.65   |   41.3%    |

# 7. Заключение
В ходе выполнения работы были успешно реализованы последовательная и параллельная версии алгоритма подсчёта частоты символа в строке.
Обе версии алгоритма прошли все функциональные тесты.
Параллельная MPI-версия демонстрирует ускорение. На 2 процессах достигнуто ускорение в 1.44 раза с эффективностью 71.8%.
При увеличении количества процессов до 4 эффективность снижается до 41.3%

# 8. Литература
1. Лекции 
2. Практические занятия 
3. Интернет

# 9. Приложение

```cpp
bool ShkrebkoMCountCharFreqMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string input_text;
  char target_char = 0;
  int total_size = 0;

  if (rank == 0) {
    input_text = std::get<0>(GetInput());
    target_char = std::get<1>(GetInput())[0];
    total_size = static_cast<int>(input_text.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&target_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    GetOutput() = 0;
    return true;
  }

  const int base = total_size / size;
  const int remainder = total_size % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    sendcounts[i] = base + (i < remainder ? 1 : 0);
    displs[i] = (i * base) + std::min(i, remainder);
  }

  int local_size = sendcounts[rank];
  std::vector<char> local_data(local_size);

  MPI_Scatterv(rank == 0 ? input_text.data() : nullptr, sendcounts.data(), displs.data(), MPI_CHAR, local_data.data(),
               sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_count = 0;
  for (char c : local_data) {
    if (c == target_char) {
      local_count++;
    }
  }

  int global_result = 0;
  MPI_Allreduce(&local_count, &global_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}
```