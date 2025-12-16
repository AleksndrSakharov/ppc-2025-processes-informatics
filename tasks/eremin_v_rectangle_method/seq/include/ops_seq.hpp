#pragma once

#include "eremin_v_rectangle_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace eremin_v_rectangle_method {

class EreminVRectangleMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit EreminVRectangleMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace eremin_v_rectangle_method
