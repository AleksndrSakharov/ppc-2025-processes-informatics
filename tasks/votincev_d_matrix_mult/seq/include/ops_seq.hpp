#pragma once

#include "task/include/task.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VotincevDMatrixMultSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace votincev_d_matrix_mult
