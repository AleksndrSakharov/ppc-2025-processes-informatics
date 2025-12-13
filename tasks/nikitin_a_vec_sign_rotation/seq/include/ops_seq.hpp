#pragma once

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinAVecSignRotationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NikitinAVecSignRotationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] static bool IsSignChange(double first_value, double second_value);
};

}  // namespace nikitin_a_vec_sign_rotation
