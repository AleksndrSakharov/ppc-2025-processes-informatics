#pragma once

#include <vector>

#include "mityaeva_d_min_v_rows_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_min_v_rows_matrix {

class MinValuesInRowsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MinValuesInRowsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

std::vector<int> ProcessLocalRows(const std::vector<int> &input, int start_row, int my_rows, int cols);
void GatherResults(int rank, int size, int rows, int rows_per_process, int remainder,
                   const std::vector<int> &local_result, std::vector<int> &output);

}  // namespace mityaeva_d_min_v_rows_matrix
