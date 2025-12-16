#pragma once

#include "egashin_k_lexicographical_check/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_lexicographical_check {

class TestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace egashin_k_lexicographical_check
