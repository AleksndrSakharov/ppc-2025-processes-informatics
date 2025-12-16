#pragma once

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_words_string_count {

class AkimovIWordsStringCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AkimovIWordsStringCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_buffer_;
  InType local_buffer_;
  int local_space_count_ = 0;
  int global_space_count_ = 0;
  int word_count_ = 0;
};
}  // namespace akimov_i_words_string_count
