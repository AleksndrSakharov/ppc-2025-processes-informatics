#pragma once

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_v_trapezoid_seq {

double Func(double x);

class TrapezoidIntegrationSequential
    : public BaseTask  // наследник BaseTask (псевдоним для ppc::task::Task<InType, OutType>)
{
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TrapezoidIntegrationSequential(
      const InType &in);  // конструктор принимает InType (псевдоним для структуры InputData)

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kutergin_v_trapezoid_seq
