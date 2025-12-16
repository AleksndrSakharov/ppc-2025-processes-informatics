#pragma once

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace shekhirev_v_char_freq_seq {

class CharFreqSequential : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit CharFreqSequential(const InType &in);

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shekhirev_v_char_freq_seq
