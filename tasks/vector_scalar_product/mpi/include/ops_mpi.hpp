#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vector_scalar_product/common/include/common.hpp"

namespace vector_scalar_product {

class VectorScalarProductMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit VectorScalarProductMpi(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rank_ = 0;
  int world_size_ = 1;
  std::vector<double> local_lhs_;
  std::vector<double> local_rhs_;
  double local_sum_ = 0.0;
  double result_ = 0.0;
};

}  // namespace vector_scalar_product
