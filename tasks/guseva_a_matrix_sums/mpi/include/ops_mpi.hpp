#pragma once

#include "guseva_a_matrix_sums/common/include/common.hpp"
#include "task/include/task.hpp"

namespace guseva_a_matrix_sums {

class GusevaAMatrixSumsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GusevaAMatrixSumsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace guseva_a_matrix_sums
