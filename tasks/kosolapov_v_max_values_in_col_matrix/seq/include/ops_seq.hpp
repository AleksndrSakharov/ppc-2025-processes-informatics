#pragma once

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

class KosolapovVMaxValuesInColMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KosolapovVMaxValuesInColMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kosolapov_v_max_values_in_col_matrix
