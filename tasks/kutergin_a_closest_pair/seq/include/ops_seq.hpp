#pragma once

#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_a_closest_pair {

class KuterginAClosestPairSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KuterginAClosestPairSEQ(const InType &in);

 private:
  std::vector<int> data_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kutergin_a_closest_pair
