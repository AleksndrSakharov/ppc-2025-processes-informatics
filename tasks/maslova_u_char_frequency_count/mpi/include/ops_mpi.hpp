#pragma once

#include "maslova_u_char_frequency_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace maslova_u_char_frequency_count {

class MaslovaUCharFrequencyCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MaslovaUCharFrequencyCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace maslova_u_char_frequency_count
