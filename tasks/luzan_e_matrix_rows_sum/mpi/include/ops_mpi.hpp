#pragma once

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luzan_e_matrix_rows_sum {

class LuzanEMatrixRowsSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LuzanEMatrixRowsSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace luzan_e_matrix_rows_sum
