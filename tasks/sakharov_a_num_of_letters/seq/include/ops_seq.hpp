#pragma once

#include "sakharov_a_num_of_letters/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sakharov_a_num_of_letters {

class SakharovANumberOfLettersSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SakharovANumberOfLettersSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sakharov_a_num_of_letters
