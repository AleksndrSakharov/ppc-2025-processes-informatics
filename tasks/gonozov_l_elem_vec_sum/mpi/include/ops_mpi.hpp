#pragma once

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gonozov_l_elem_vec_sum {

class GonozovLElemVecSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GonozovLElemVecSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int vector_size_ = 0;
};

}  // namespace gonozov_l_elem_vec_sum
