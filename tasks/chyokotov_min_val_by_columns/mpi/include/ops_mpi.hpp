#pragma once

#include "chyokotov_min_val_by_columns/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chyokotov_min_val_by_columns {

class ChyokotovMinValByColumnsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChyokotovMinValByColumnsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chyokotov_min_val_by_columns
