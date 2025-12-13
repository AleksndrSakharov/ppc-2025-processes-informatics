#pragma once

#include <cstddef>
#include <vector>

#include "papulina_y_simple_iteration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace papulina_y_simple_iteration {

class PapulinaYSimpleIterationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PapulinaYSimpleIterationMPI(const InType &in);

 private:
  static double CalculateNormB(const std::vector<double> &a, size_t n);
  void CalculateGatherParameters(std::vector<int> &proc_count_elemts_x, std::vector<int> &x_displs, int rows_for_proc,
                                 int remainder) const;
  void PrepareLocalMatrices(std::vector<double> &local_b_matrix, std::vector<double> &local_d, int start_row,
                            int local_rows_count, int n);
  static bool FindAndSwapRow(std::vector<double> &tmp, size_t i, size_t n);
  static bool DetermChecking(const std::vector<double> &a, const size_t &n);
  static bool DiagonalDominance(const std::vector<double> &a, const size_t &n);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int procNum_ = 0;
  std::vector<double> A_;
  std::vector<double> local_a_;
  std::vector<double> local_b_;
  std::vector<double> b_;
  std::vector<double> result_;
  size_t n_ = 0;
  unsigned int steps_count_ = 100000;
  double eps_ = 1e-7;
};

}  // namespace papulina_y_simple_iteration
