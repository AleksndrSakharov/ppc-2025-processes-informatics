#pragma once

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkrebko_m_count_char_freq {

class ShkrebkoMCountCharFreqSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShkrebkoMCountCharFreqSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shkrebko_m_count_char_freq
