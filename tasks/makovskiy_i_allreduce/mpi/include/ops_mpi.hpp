#pragma once

#include <vector>

#include "makovskiy_i_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makovskiy_i_allreduce {

class TestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_input_;
};

}  // namespace makovskiy_i_allreduce
