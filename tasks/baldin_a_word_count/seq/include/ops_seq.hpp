#pragma once

#include "baldin_a_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baldin_a_word_count {

class BaldinAWordCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BaldinAWordCountSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baldin_a_word_count
