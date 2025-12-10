#pragma once

#include "baldin_a_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baldin_a_word_count {

class BaldinAWordCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaldinAWordCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baldin_a_word_count
