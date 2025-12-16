#pragma once

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rozenberg_a_matrix_column_sum {

class RozenbergAMatrixColumnSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RozenbergAMatrixColumnSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rozenberg_a_matrix_column_sum
