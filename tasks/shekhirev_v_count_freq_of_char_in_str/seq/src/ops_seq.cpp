#include "../include/ops_seq.hpp"

#include "../../common/include/common.hpp"

namespace shekhirev_v_char_freq_seq {

CharFreqSequential::CharFreqSequential(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool CharFreqSequential::ValidationImpl() {
  return true;
}

bool CharFreqSequential::PreProcessingImpl() {
  return true;
}

bool CharFreqSequential::RunImpl() {
  const auto &str = GetInput().str;
  char target = GetInput().target;

  int count = 0;
  for (char c : str) {
    if (c == target) {
      count++;
    }
  }

  GetOutput() = count;
  return true;
}

bool CharFreqSequential::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_char_freq_seq
