#pragma once

#include "task/include/task.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatingValuesSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VotincevDAlternatingValuesSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // == мои дополнительные функции ==
  static bool IsSignChange(const double &a, const double &b);
  // ================================
};

}  // namespace votincev_d_alternating_values
