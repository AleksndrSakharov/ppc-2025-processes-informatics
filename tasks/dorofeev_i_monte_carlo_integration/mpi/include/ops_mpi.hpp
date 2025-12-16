#pragma once

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

class DorofeevIMonteCarloIntegrationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DorofeevIMonteCarloIntegrationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dorofeev_i_monte_carlo_integration_processes
