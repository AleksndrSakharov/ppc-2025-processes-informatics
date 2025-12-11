#pragma once

#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

class MinValueMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MinValueMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ProcessRankZero(std::vector<int> &local_min_values);
  void GatherResults(const std::vector<int> &local_min_values);

  static void ProcessWorkerRank(std::vector<int> &local_min_values);
};

}  // namespace makovskiy_i_min_value_in_matrix_rows
