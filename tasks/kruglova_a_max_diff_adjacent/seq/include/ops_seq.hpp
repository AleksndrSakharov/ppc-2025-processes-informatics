#pragma once

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kruglova_a_max_diff_adjacent {

class KruglovaAMaxDiffAdjacentSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KruglovaAMaxDiffAdjacentSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kruglova_a_max_diff_adjacent
