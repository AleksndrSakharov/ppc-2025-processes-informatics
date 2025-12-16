#pragma once

#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rozenberg_a_matrix_column_sum {

class RozenbergAMatrixColumnSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RozenbergAMatrixColumnSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<int> flat_;
};

}  // namespace rozenberg_a_matrix_column_sum
