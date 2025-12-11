#pragma once

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_min_val_rows_matrix {

class LevonychevIMinValRowsMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LevonychevIMinValRowsMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace levonychev_i_min_val_rows_matrix
