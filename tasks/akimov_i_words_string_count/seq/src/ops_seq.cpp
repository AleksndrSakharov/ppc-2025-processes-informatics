#include "akimov_i_words_string_count/seq/include/ops_seq.hpp"

#include <cctype>

#include "akimov_i_words_string_count/common/include/common.hpp"

namespace akimov_i_words_string_count {

AkimovIWordsStringCountSEQ::AkimovIWordsStringCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIWordsStringCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool AkimovIWordsStringCountSEQ::PreProcessingImpl() {
  input_buffer_ = GetInput();
  word_count_ = 0;
  space_count_ = 0;
  return true;
}

bool AkimovIWordsStringCountSEQ::RunImpl() {
  if (input_buffer_.empty()) {
    word_count_ = 0;
    return true;
  }

  for (char c : input_buffer_) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      ++space_count_;
    }
  }

  bool in_word = false;
  word_count_ = 0;
  for (char c : input_buffer_) {
    if (std::isspace(static_cast<unsigned char>(c)) == 0 && !in_word) {
      in_word = true;
      ++word_count_;
    } else if (std::isspace(static_cast<unsigned char>(c)) != 0 && in_word) {
      in_word = false;
    }
  }

  return true;
}

bool AkimovIWordsStringCountSEQ::PostProcessingImpl() {
  GetOutput() = word_count_;
  return true;
}

}  // namespace akimov_i_words_string_count
