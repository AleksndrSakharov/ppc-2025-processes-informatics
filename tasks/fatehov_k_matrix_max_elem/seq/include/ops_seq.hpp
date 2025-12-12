#pragma once

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fatehov_k_matrix_max_elem {

class FatehovKMatrixMaxElemSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit FatehovKMatrixMaxElemSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static const int kMaxRows = 10000;
  static const int kMaxCols = 10000;
  static const int kMaxMatrixSize = 100000000;
};

}  // namespace fatehov_k_matrix_max_elem
