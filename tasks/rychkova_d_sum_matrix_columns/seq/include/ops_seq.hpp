#pragma once

#include "rychkova_d_sum_matrix_columns/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rychkova_d_sum_matrix_columns {

class RychkovaDSumMatrixColumnsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit RychkovaDSumMatrixColumnsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rychkova_d_sum_matrix_columns
