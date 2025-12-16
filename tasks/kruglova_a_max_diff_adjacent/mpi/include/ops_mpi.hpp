#pragma once
#include <vector>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kruglova_a_max_diff_adjacent {

class KruglovaAMaxDiffAdjacentMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KruglovaAMaxDiffAdjacentMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static float LocalMaxDiff(const std::vector<float> &local_vec);
};

}  // namespace kruglova_a_max_diff_adjacent
