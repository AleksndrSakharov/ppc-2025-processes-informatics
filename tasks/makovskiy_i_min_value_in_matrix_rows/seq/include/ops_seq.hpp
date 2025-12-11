#pragma once

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

class MinValueSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MinValueSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makovskiy_i_min_value_in_matrix_rows
