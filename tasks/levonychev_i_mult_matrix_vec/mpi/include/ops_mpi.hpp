#pragma once

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_mult_matrix_vec {

class LevonychevIMultMatrixVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LevonychevIMultMatrixVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace levonychev_i_mult_matrix_vec
