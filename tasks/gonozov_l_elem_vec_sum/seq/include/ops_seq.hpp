#pragma once

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gonozov_l_elem_vec_sum {

class GonozovLElemVecSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GonozovLElemVecSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gonozov_l_elem_vec_sum
