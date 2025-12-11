#pragma once

#include "task/include/task.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByColumnsMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZeninASumValuesByColumnsMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zenin_a_sum_values_by_columns_matrix
