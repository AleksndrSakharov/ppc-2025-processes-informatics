#pragma once

#include <vector>

#include "egashin_k_lexicographical_check/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_lexicographical_check {

class TestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateDistribution(int size, int min_len, std::vector<int> &counts, std::vector<int> &displs);
  static int CompareLocal(const std::vector<char> &s1, const std::vector<char> &s2, int count);
  static bool GetFinalDecision(const std::vector<int> &global_results, int s1_len, int s2_len);
};

}  // namespace egashin_k_lexicographical_check
