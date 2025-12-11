#pragma once

#include "task/include/task.hpp"
#include "zavyalov_a_scalar_product/common/include/common.hpp"

namespace zavyalov_a_scalar_product {

class ZavyalovAScalarProductMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZavyalovAScalarProductMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zavyalov_a_scalar_product
