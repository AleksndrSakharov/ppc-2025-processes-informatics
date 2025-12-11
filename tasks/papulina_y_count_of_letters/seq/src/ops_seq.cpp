#include "papulina_y_count_of_letters/seq/include/ops_seq.hpp"

#include <cctype>

#include "papulina_y_count_of_letters/common/include/common.hpp"

namespace papulina_y_count_of_letters {

PapulinaYCountOfLettersSEQ::PapulinaYCountOfLettersSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}
int PapulinaYCountOfLettersSEQ::CountOfLetters(const char *s, const int &n) {
  int k = 0;
  if (n <= 0) {
    return 0;
  }
  for (int i = 0; i < n; i++) {
    unsigned char c = s[i];
    if (isalpha(c) != 0) {
      k++;
    }
  }
  return k;
}
bool PapulinaYCountOfLettersSEQ::ValidationImpl() {
  return true;
}

bool PapulinaYCountOfLettersSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return GetOutput() == 0;
}

bool PapulinaYCountOfLettersSEQ::RunImpl() {
  GetOutput() = CountOfLetters(GetInput().data(), static_cast<int>(GetInput().size()));
  return true;
}

bool PapulinaYCountOfLettersSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace papulina_y_count_of_letters
