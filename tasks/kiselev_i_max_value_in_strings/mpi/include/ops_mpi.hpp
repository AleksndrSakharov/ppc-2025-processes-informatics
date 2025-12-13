#pragma once

#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevITestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KiselevITestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void DistributeRowLengths(const std::vector<std::vector<int>> &matrix, int total_rows, int world_rank,
                                   int world_size, std::vector<int> &local_row_lengths, std::vector<int> &len_counts,
                                   std::vector<int> &len_displs);
  static void DistributeValues(const std::vector<std::vector<int>> &matrix, int world_rank, int world_size,
                               const std::vector<int> &len_counts, const std::vector<int> &len_displs,
                               std::vector<int> &local_values);
  static void ComputeLocalMax(const std::vector<int> &local_values, const std::vector<int> &local_row_lengths,
                              std::vector<int> &local_result);
};

}  // namespace kiselev_i_max_value_in_strings
