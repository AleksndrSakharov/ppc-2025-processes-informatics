#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VotincevDMatrixMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // ==============================
  // мои дополнительные функции ===

  static void SendData(int k, int proc_rank, int process_n, std::vector<int> &my_range, std::vector<int> &ranges,
                       std::vector<double> &local_matrix, std::vector<double> &matrix_a);

  static void MatrixPartMult(int param_k, int param_n, std::vector<double> &local_matrix,
                             const std::vector<double> &matrix_b);

  static std::vector<double> SeqMatrixMult(int param_m, int param_n, int param_k, std::vector<double> &matrix_a,
                                           std::vector<double> &matrix_b);
};

}  // namespace votincev_d_matrix_mult
