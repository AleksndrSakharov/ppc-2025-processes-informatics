# Подсчет частоты символа в строке

- Student: Маслова Ульяна Александровна, group 3823Б1ФИ2
- Technology: SEQ | MPI
- Variant: 23

## 1. Introduction
Проблема: Последовательный подсчет частоты символов в строках большого размера является медленным.
Задача: Ускорить этот процесс с помощью параллельных вычислений на MPI.
Ожидаемый результат: Значительное сокращение времени выполнения по сравнению с последовательной версией.

## 2. Problem Statement
Нужно найти число вхождений символа input_char в строке input_str.
- InPut: Пара (std::string, char).
- OutPut: Целое число (size_t).

## 3. Baseline Algorithm (Sequential)
Проход по строке в цикле с увеличением счетчика при нахождении искомого символа. Алгоритм имеет линейную временную сложность O(N), где N — длина строки. 

## 4. Parallelization Scheme
Процесс с рангом 0 делит исходную строку на P (число процессов) частей. Далее ранг 0 рассылает каждому процессу его фрагмент строки. Каждый процесс независимо считает символы в своей части. Локальные счетчики суммируются на ранге 0.

## 5. Implementation Details
- common: Определяет общие типы данных (InType, OutType).
- seq: Содержит простую последовательную реализацию алгоритма.
- mpi: Содержит параллельную MPI-реализацию алгоритма.
- tests: Включает два набора тестов: functional для проверки корректности и performance для замера скорости.
Максимальная длинна строки, которая может быть обработана программой - 2<sup>31</sup> - 1, что составляет 2 147 483 647 символов.

## 6. Experimental Setup
- Аппаратное обеспечение: AMD Ryzen 7 7840HS (8 ядер, 16 логических процессоров, базовая частота 3,80 ГГц)
- ОЗУ — 16 ГБ 
- Операционная система: Windows 11
- Компилятор: g++
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness
Корректность проверялась на строках различной длинны, а также различного типа (пустые, из одного слова, только из букв, смешанные и т.д.)

### 7.2 Performance

Тест на данных, состоящих из 100 000 000 символов:

| Mode | Count | Time, s   | Speedup | Efficiency |
|------|-------|-----------|---------|------------|
| seq  | 1     | 0.07242   | 1.00    | N/A        |
| mpi  | 2     | 0.04439   | 1.63    | 81.5%      |
| mpi  | 4     | 0.03100   | 2.34    | 58.5%      |
| mpi  | 8     | 0.03050   | 2.37    | 29.6%      |

## 8. Conclusions
Мы видим значительное повышение производительности. С увеличением числа процессов время выполнения сокращается, в связи с этим мы имеем ускорение 2.34 на 4 процессах. В свою очередь эффективность падает с ростом числа процессов, так как накладные расходы на коммуникацию MPI на 8 процессах начинают перевешивать выгоду от параллелизма.

## 9. References
1. Лекции и практики курса "Параллельное программирование"

## Appendix (Optional)
```cpp
bool MaslovaUCharFrequencyCountMPI::RunImpl() {
  int rank = 0;
  int proc_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // id процесса
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);  // количество процессов

  std::string input_string;
  char input_char = 0;
  size_t input_str_size = 0;

  if (rank == 0) {
    input_string = GetInput().first;
    input_char = GetInput().second;
    input_str_size = input_string.size();  // получили данные
  }

  uint64_t size_for_mpi = 0;
  if (rank == 0) {
    size_for_mpi = static_cast<uint64_t>(input_str_size);  // явное приведение перед передачей
  }

  MPI_Bcast(&size_for_mpi, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);  // отправляем размер строки

  if (rank != 0) {
    input_str_size = static_cast<size_t>(size_for_mpi);  // возращаем обратно для удобного использования в дальнейшем
  }

  if (input_str_size == 0) {
    GetOutput() = 0;  // ставим для всех процессов
    return true;
  }

  MPI_Bcast(&input_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);  // отправляем нужный символ

  std::vector<int> send_counts(proc_size);  // здесь размеры всех порций
  std::vector<int> displs(proc_size);       // смещения
  if (rank == 0) {
    size_t part = input_str_size / proc_size;
    size_t rem = input_str_size % proc_size;
    for (size_t i = 0; std::cmp_less(i, proc_size); ++i) {
      send_counts[i] = static_cast<int>(part + (i < rem ? 1 : 0));  // общий размер, включающий остаток, если он входит
    }
    displs[0] = 0;
    for (size_t i = 1; std::cmp_less(i, proc_size); ++i) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  MPI_Bcast(send_counts.data(), proc_size, MPI_INT, 0, MPI_COMM_WORLD);  // отправляем размеры порций
  std::vector<char> local_str(send_counts[rank]);
  MPI_Scatterv((rank == 0) ? input_string.data() : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_str.data(), static_cast<int>(local_str.size()), MPI_CHAR, 0, MPI_COMM_WORLD  // распределяем данные
  );

  size_t local_count = std::count(local_str.begin(), local_str.end(), input_char);
  auto local_count_for_mpi = static_cast<uint64_t>(local_count);
  uint64_t global_count = 0;
  MPI_Allreduce(&local_count_for_mpi, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);  // собрали данные со всех процессов

  GetOutput() = static_cast<size_t>(global_count);  // вывели результат, при этом приведя его к нужному нам типу

  return true;
}

bool MaslovaUCharFrequencyCountMPI::PostProcessingImpl() {
  return true;
}
```