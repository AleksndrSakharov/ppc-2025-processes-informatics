#pragma once

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ashihmin_d_sum_of_elem {

class AshihminDElemVecsSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AshihminDElemVecsSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ashihmin_d_sum_of_elem
