#pragma once

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gutyansky_a_matrix_column_sum {

class GutyanskyAMatrixColumnSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GutyanskyAMatrixColumnSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gutyansky_a_matrix_column_sum
