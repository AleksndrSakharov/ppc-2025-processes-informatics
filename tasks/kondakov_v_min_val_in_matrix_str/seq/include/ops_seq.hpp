#pragma once

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_min_val_in_matrix_str {

class KondakovVMinValMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KondakovVMinValMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kondakov_v_min_val_in_matrix_str
