#pragma once

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "task/include/task.hpp"

namespace papulina_y_count_of_letters {

class PapulinaYCountOfLettersSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PapulinaYCountOfLettersSEQ(const InType &in);

 private:
  static int CountOfLetters(const char *s, const int &n);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace papulina_y_count_of_letters
