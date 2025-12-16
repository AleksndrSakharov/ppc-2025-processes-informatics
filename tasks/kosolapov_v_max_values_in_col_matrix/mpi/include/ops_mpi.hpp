#pragma once

#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

class KosolapovVMaxValuesInColMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KosolapovVMaxValuesInColMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static std::vector<int> CalculateLocalMax(const std::vector<std::vector<int>> &matrix, int columns);
  void DistributeDataFromRoot(std::vector<std::vector<int>> &local_matrix, int start, int local_rows, int columns,
                              int processes_count, int rows_per_proc, int remainder);
};

}  // namespace kosolapov_v_max_values_in_col_matrix
