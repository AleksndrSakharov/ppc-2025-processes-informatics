#pragma once

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "task/include/task.hpp"

namespace papulina_y_count_of_letters {

class PapulinaYCountOfLettersMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PapulinaYCountOfLettersMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static int CountOfLetters(const char *s, const int &n);

  int procNum_ = 0;
};

}  // namespace papulina_y_count_of_letters
