#pragma once

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gasenin_l_lex_dif {

class GaseninLLexDifMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GaseninLLexDifMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gasenin_l_lex_dif
