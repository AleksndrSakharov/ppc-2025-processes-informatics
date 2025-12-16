#include "papulina_y_simple_iteration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "papulina_y_simple_iteration/common/include/common.hpp"

namespace papulina_y_simple_iteration {

PapulinaYSimpleIterationMPI::PapulinaYSimpleIterationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>(0);
  MPI_Comm_size(MPI_COMM_WORLD, &procNum_);
}
bool PapulinaYSimpleIterationMPI::DiagonalDominance(const std::vector<double> &a, const size_t &n) {
  bool flag = true;
  for (size_t i = 0; i < n; i++) {
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
      if (j != i) {
        sum += abs(a[(i * n) + j]);
      }
    }
    if (sum > abs(a[(i * n) + i])) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool PapulinaYSimpleIterationMPI::DetermChecking(const std::vector<double> &a, const size_t &n) {
  std::vector<double> tmp = a;

  for (size_t i = 0; i < n; i++) {
    if (std::fabs(tmp[(i * n) + i]) < 1e-10) {
      if (!FindAndSwapRow(tmp, i, n)) {
        return false;
      }
    }
    double pivot = tmp[(i * n) + i];
    for (size_t j = i; j < n; j++) {
      tmp[(i * n) + j] /= pivot;
    }
    for (size_t j = i + 1; j < n; j++) {
      double factor = tmp[(j * n) + i];
      for (size_t k = i; k < n; k++) {
        tmp[(j * n) + k] -= tmp[(i * n) + k] * factor;
      }
    }
  }

  return true;
}
bool PapulinaYSimpleIterationMPI::FindAndSwapRow(std::vector<double> &tmp, size_t i, size_t n) {
  for (size_t j = i + 1; j < n; j++) {
    if (std::fabs(tmp[(j * n) + i]) > 1e-10) {
      for (size_t k = i; k < n; k++) {
        std::swap(tmp[(i * n) + k], tmp[(j * n) + k]);
      }
      return true;
    }
  }
  return false;
}
bool PapulinaYSimpleIterationMPI::ValidationImpl() {
  int flag = 1;
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    size_t n = std::get<0>(GetInput());
    const auto &a_matrix = std::get<1>(GetInput());

    if ((n < 1) || (!DetermChecking(a_matrix, n) || (!DiagonalDominance(a_matrix, n)))) {
      flag = 0;  // валидация не прошла
    } else {
      double norm_b = CalculateNormB(a_matrix, n);
      if (norm_b >= 1.0) {
        std::cout << "WARNING: sufficient condition for convergence may not hold (norm_b = " << norm_b << " >= 1)\n";
      }
    }
  }

  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return (flag == 1);
}
double PapulinaYSimpleIterationMPI::CalculateNormB(const std::vector<double> &a, size_t n) {
  double max_row_sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    double diag = a[(i * n) + i];
    double row_sum = 0.0;

    for (size_t j = 0; j < n; j++) {
      if (j != i) {
        row_sum += std::abs(a[(i * n) + j] / diag);
      }
    }
    max_row_sum = std::max(row_sum, max_row_sum);
  }

  return max_row_sum;
}
bool PapulinaYSimpleIterationMPI::PreProcessingImpl() {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    n_ = static_cast<size_t>(std::get<0>(GetInput()));
    A_.assign(n_ * n_, 0.0);
    b_.assign(n_, 0.0);

    std::copy(std::get<1>(GetInput()).data(), std::get<1>(GetInput()).data() + (n_ * n_), A_.data());
    std::copy(std::get<2>(GetInput()).data(), std::get<2>(GetInput()).data() + n_, b_.data());
  }

  MPI_Bcast(&n_, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  int rows_for_proc = static_cast<int>(n_) / procNum_;
  int remainder = static_cast<int>(n_) % procNum_;

  int start_row = (proc_rank * rows_for_proc) + std::min(proc_rank, remainder);
  int last_row = start_row + rows_for_proc + (proc_rank < remainder ? 1 : 0);
  int local_rows_count = last_row - start_row;

  local_a_.resize(local_rows_count * n_);
  local_b_.resize(local_rows_count);

  std::vector<int> proc_count_elements_a(procNum_, 0);
  std::vector<int> proc_count_elements_b(procNum_, 0);
  std::vector<int> a_displs(procNum_, 0);
  std::vector<int> b_displs(procNum_, 0);

  for (int i = 0; i < procNum_; i++) {
    int start = (i * rows_for_proc) + std::min(i, remainder);
    int end = start + rows_for_proc + (i < remainder ? 1 : 0);
    int count = end - start;

    proc_count_elements_a[i] = count * static_cast<int>(n_);
    proc_count_elements_b[i] = count;

    if (i > 0) {
      a_displs[i] = a_displs[i - 1] + proc_count_elements_a[i - 1];
      b_displs[i] = b_displs[i - 1] + proc_count_elements_b[i - 1];
    }
  }

  MPI_Scatterv((proc_rank == 0) ? A_.data() : nullptr, proc_count_elements_a.data(), a_displs.data(), MPI_DOUBLE,
               local_a_.data(), local_rows_count * static_cast<int>(n_), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv((proc_rank == 0) ? b_.data() : nullptr, proc_count_elements_b.data(), b_displs.data(), MPI_DOUBLE,
               local_b_.data(), local_rows_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool PapulinaYSimpleIterationMPI::RunImpl() {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  int rows_for_proc = static_cast<int>(n_) / procNum_;
  int remainder = static_cast<int>(n_) % procNum_;

  int start_row = (proc_rank * rows_for_proc) + std::min(proc_rank, remainder);
  int last_row = start_row + rows_for_proc + (proc_rank < remainder ? 1 : 0);
  int local_rows_count = last_row - start_row;
  std::vector<double> local_b_matrix(local_rows_count * n_, 0);
  std::vector<double> local_d(local_rows_count, 0);

  PrepareLocalMatrices(local_b_matrix, local_d, start_row, local_rows_count, static_cast<int>(n_));

  std::vector<double> x(n_, 0.0);
  std::vector<double> x_new(n_, 0.0);

  double diff = 0.0;

  for (unsigned int step = 0; step < steps_count_; step++) {
    std::vector<double> local_x_new(local_rows_count, 0.0);
    for (size_t i = 0; std::cmp_less(i, local_rows_count); i++) {
      for (size_t j = 0; j < n_; j++) {
        local_x_new[i] += local_b_matrix[(i * n_) + j] * x[j];
      }
      local_x_new[i] += local_d[i];
    }

    std::vector<int> proc_count_elemts_x(procNum_, 0);
    std::vector<int> x_displs(procNum_, 0);
    CalculateGatherParameters(proc_count_elemts_x, x_displs, rows_for_proc, remainder);
    // MPI_Gatherv(local_x_new.data(), local_rows_count, MPI_DOUBLE, x_new.data(), proc_count_elemts_x.data(),
    // x_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Bcast(x_new.data(), static_cast<int>(n_), MPI_DOUBLE, 0,
    // MPI_COMM_WORLD);
    MPI_Allgatherv(local_x_new.data(), local_rows_count, MPI_DOUBLE, x_new.data(), proc_count_elemts_x.data(),
                   x_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    double local_sum_for_norm = 0.0;
    for (int i = 0; i < local_rows_count; i++) {
      int global_i = start_row + i;
      double diff_i = x_new[global_i] - x[global_i];
      local_sum_for_norm += diff_i * diff_i;
    }

    MPI_Allreduce(&local_sum_for_norm, &diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    diff = std::sqrt(diff);
    if (diff < eps_) {
      x = x_new;
      break;
    }
    x = x_new;
  }
  GetOutput() = x;
  return true;
}
bool PapulinaYSimpleIterationMPI::PostProcessingImpl() {
  return true;
}
void PapulinaYSimpleIterationMPI::CalculateGatherParameters(std::vector<int> &proc_count_elemts_x,
                                                            std::vector<int> &x_displs, int rows_for_proc,
                                                            int remainder) const {
  // if (proc_rank != 0) {
  //   return;
  // }
  for (int i = 0; i < procNum_; i++) {
    unsigned int start = (i * rows_for_proc) + std::min(i, remainder);
    unsigned int end = start + rows_for_proc + (i < remainder ? 1 : 0);
    unsigned int count = end - start;
    proc_count_elemts_x[i] = static_cast<int>(count);

    if (i > 0) {
      x_displs[i] = x_displs[i - 1] + proc_count_elemts_x[i - 1];
    }
  }
}
void PapulinaYSimpleIterationMPI::PrepareLocalMatrices(std::vector<double> &local_b_matrix,
                                                       std::vector<double> &local_d, int start_row,
                                                       int local_rows_count, int n) {
  for (int i = 0; i < local_rows_count; i++) {
    int global_i = start_row + i;
    double diag = local_a_[(i * n) + global_i];
    double inv_diag = 1.0 / diag;
    for (int j = 0; j < n; j++) {
      if (j != global_i) {
        local_b_matrix[(i * n) + j] = -local_a_[(i * n) + j] * inv_diag;
      }
    }
    local_d[i] = local_b_[i] * inv_diag;
  }
}
}  // namespace papulina_y_simple_iteration
