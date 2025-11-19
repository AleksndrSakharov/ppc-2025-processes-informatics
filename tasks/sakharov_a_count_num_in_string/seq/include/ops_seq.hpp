#pragma once

#include "sakharov_a_count_num_in_string/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sakharov_a_count_num_in_string {

class SakharovATestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SakharovATestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sakharov_a_count_num_in_string
