#pragma once

#include "task/include/task.hpp"
#include "telnov_counting_the_frequency/common/include/common.hpp"

namespace telnov_counting_the_frequency {

class TelnovCountingTheFrequencySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TelnovCountingTheFrequencySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace telnov_counting_the_frequency
