#pragma once

#include "nikitina_v_trans_all_one_distrib/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitina_v_trans_all_one_distrib {

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
};

}  // namespace nikitina_v_trans_all_one_distrib
