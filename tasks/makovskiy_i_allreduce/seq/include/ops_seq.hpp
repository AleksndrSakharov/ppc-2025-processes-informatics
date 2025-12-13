#pragma once

#include "makovskiy_i_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makovskiy_i_allreduce {

class TestTaskSEQ : public BaseTask {
 public:
  // для покрытия
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makovskiy_i_allreduce
