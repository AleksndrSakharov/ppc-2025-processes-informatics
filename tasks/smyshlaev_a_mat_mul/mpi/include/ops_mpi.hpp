#pragma once

#include <vector>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_mat_mul {

class SmyshlaevAMatMulMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SmyshlaevAMatMulMPI(const InType &in);

 private:
  std::vector<double> mat_b_transposed_;
  bool RunSequential();
  void BroadcastDimensions(int &rows_a, int &cols_a, int &cols_b);
  static void RingShiftAlgorithm(int rank, int size, int my_rows_a, int num_cols_a, int num_cols_b,
                                 const std::vector<int> &counts_b, const std::vector<int> &disps_b,
                                 std::vector<double> &local_a, std::vector<double> &local_b,
                                 std::vector<double> &local_c);

  void GatherAndBroadcastResults(int rank, int size, int rows_a, int cols_a, int cols_b,
                                 const std::vector<int> &counts_a, const std::vector<double> &local_c);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_mat_mul
