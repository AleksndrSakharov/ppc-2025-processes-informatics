#pragma once

#include "pikhotskiy_r_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_elem_vec_sum {
class PikhotskiyRElemVecSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PikhotskiyRElemVecSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace pikhotskiy_r_elem_vec_sum
