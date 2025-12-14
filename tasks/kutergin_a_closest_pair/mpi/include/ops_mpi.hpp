#pragma once

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_a_closest_pair {

class KuterginAClosestPairMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KuterginAClosestPairMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kutergin_a_closest_pair
