#pragma once

#include "romanov_a_integration_rect_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_a_integration_rect_method {

class RomanovAIntegrationRectMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RomanovAIntegrationRectMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace romanov_a_integration_rect_method
