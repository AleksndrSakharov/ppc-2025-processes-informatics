#pragma once

#include <cstddef>
#include <vector>

#include "papulina_y_simple_iteration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace papulina_y_simple_iteration {

class PapulinaYSimpleIterationSEQ : public BaseTask {
 public:
  static double CalculateNormB(const std::vector<double> &a, size_t n);
  static bool GetDetermCheckingResult(const std::vector<double> &a, const size_t &n);
  static bool GetDiagonalDominanceResult(const std::vector<double> &a, const size_t &n);
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PapulinaYSimpleIterationSEQ(const InType &in);

 private:
  static bool FindAndSwapRow(std::vector<double> &tmp, size_t i, size_t n);
  static bool DetermChecking(const std::vector<double> &a, const size_t &n);
  static bool DiagonalDominance(const std::vector<double> &a, const size_t &n);
  static void ComputeNewX(const std::vector<double> &b_matrix, const std::vector<double> &d,
                          const std::vector<double> &x, std::vector<double> &x_new, size_t n);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> result_;
  size_t n_ = 0;
  size_t steps_count_ = 100000;
  double eps_ = 1e-7;
};

}  // namespace papulina_y_simple_iteration
