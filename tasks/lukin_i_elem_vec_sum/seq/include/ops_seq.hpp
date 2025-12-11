#pragma once

#include "lukin_i_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_elem_vec_sum {

class LukinIElemVecSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LukinIElemVecSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int vec_size_ = 0;
};
}  // namespace lukin_i_elem_vec_sum
