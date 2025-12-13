#pragma once

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_words_string_count {

class AkimovIWordsStringCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit AkimovIWordsStringCountSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_buffer_;
  int word_count_ = 0;
  int space_count_ = 0;
};

}  // namespace akimov_i_words_string_count
