#pragma once

#include "lifanov_k_adj_inv_count_restore/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lifanov_k_adj_inv_count_restore {

class LifanovKAdjacentInversionCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit LifanovKAdjacentInversionCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lifanov_k_adj_inv_count_restore
