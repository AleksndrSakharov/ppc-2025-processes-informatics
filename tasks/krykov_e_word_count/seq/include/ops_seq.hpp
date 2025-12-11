#pragma once

#include "krykov_e_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krykov_e_word_count {

class KrykovEWordCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrykovEWordCountSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krykov_e_word_count
