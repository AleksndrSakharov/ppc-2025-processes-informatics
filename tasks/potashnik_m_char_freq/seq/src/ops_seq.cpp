#include "potashnik_m_char_freq/seq/include/ops_seq.hpp"

#include <string>

#include "potashnik_m_char_freq/common/include/common.hpp"

namespace potashnik_m_char_freq {

PotashnikMCharFreqSEQ::PotashnikMCharFreqSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PotashnikMCharFreqSEQ::ValidationImpl() {
  return !std::get<0>(GetInput()).empty();
}

bool PotashnikMCharFreqSEQ::PreProcessingImpl() {
  return true;
}

bool PotashnikMCharFreqSEQ::RunImpl() {
  auto &input = GetInput();
  std::string str = std::get<0>(input);
  char chr = std::get<1>(input);

  int string_size = static_cast<int>(str.size());
  int res = 0;
  for (int i = 0; i < string_size; i++) {
    if (str[i] == chr) {
      res++;
    }
  }
  GetOutput() = res;

  return true;
}

bool PotashnikMCharFreqSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_char_freq
