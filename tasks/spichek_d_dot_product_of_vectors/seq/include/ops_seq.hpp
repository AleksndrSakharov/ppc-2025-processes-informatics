#pragma once

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "task/include/task.hpp"

namespace spichek_d_dot_product_of_vectors {

class SpichekDDotProductOfVectorsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SpichekDDotProductOfVectorsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace spichek_d_dot_product_of_vectors
