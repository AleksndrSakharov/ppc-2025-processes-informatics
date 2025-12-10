#pragma once

#include <cstddef>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

class RomanovaVMinByMatrixRowsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RomanovaVMinByMatrixRowsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  size_t n_{0};
  size_t m_{0};
  InType in_data_;
  OutType res_;
};

}  // namespace romanova_v_min_by_matrix_rows_processes
