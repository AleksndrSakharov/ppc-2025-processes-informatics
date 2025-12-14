#pragma once

#include "baldin_a_gauss_filter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baldin_a_gauss_filter {

class BaldinAGaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaldinAGaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baldin_a_gauss_filter
