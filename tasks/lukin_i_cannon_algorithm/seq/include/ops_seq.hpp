#pragma once

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_cannon_algorithm {

class LukinICannonAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LukinICannonAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int size_ = 0;
};

}  // namespace lukin_i_cannon_algorithm
