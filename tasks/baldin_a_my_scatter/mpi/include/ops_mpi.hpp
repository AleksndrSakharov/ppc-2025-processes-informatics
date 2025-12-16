#pragma once

#include "baldin_a_my_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baldin_a_my_scatter {

class BaldinAMyScatterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaldinAMyScatterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baldin_a_my_scatter
