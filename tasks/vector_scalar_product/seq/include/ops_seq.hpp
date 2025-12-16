#pragma once

#include "task/include/task.hpp"
#include "vector_scalar_product/common/include/common.hpp"

namespace vector_scalar_product {

class VectorScalarProductSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit VectorScalarProductSeq(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double partial_sum_ = 0.0;
};

}  // namespace vector_scalar_product
