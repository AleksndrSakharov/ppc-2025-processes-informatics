#pragma once

#include "potashnik_m_char_freq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace potashnik_m_char_freq {

class PotashnikMCharFreqMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PotashnikMCharFreqMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace potashnik_m_char_freq
