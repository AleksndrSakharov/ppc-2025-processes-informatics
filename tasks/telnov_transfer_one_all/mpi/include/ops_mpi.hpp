#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "telnov_transfer_one_all/common/include/common.hpp"

namespace telnov_transfer_one_all {
template <typename T>
class TelnovTransferOneAllMPI : public BaseTask<T> {
 public:
  using InType = std::vector<T>;
  using OutType = InType;
  static ppc::task::TypeOfTask GetStaticTypeOfTask([[maybe_unused]] bool dummy = false) {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TelnovTransferOneAllMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace telnov_transfer_one_all
