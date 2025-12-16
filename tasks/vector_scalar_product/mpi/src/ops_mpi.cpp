#include "vector_scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "vector_scalar_product/common/include/common.hpp"

namespace vector_scalar_product {
namespace {
std::vector<int> BuildCounts(int total, int parts) {
  std::vector<int> counts(parts, 0);
  const int base = total / parts;
  int remainder = total % parts;
  for (int i = 0; i < parts; ++i) {
    counts[i] = base + (remainder > 0 ? 1 : 0);
    if (remainder > 0) {
      --remainder;
    }
  }
  return counts;
}

std::vector<int> BuildDisplacements(const std::vector<int> &counts) {
  std::vector<int> displs(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  return displs;
}
}  // namespace

VectorScalarProductMpi::VectorScalarProductMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool VectorScalarProductMpi::ValidationImpl() {
  const auto &lhs = GetInput().lhs;
  const auto &rhs = GetInput().rhs;
  return !lhs.empty() && lhs.size() == rhs.size();
}

bool VectorScalarProductMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  const auto &lhs = GetInput().lhs;
  const auto &rhs = GetInput().rhs;
  int global_size = 0;
  if (rank_ == 0) {
    global_size = static_cast<int>(lhs.size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Prepare scattering meta information
  auto counts = BuildCounts(global_size, world_size_);
  auto displs = BuildDisplacements(counts);
  const int local_count = counts[rank_];

  local_lhs_.assign(static_cast<std::size_t>(local_count), 0.0);
  local_rhs_.assign(static_cast<std::size_t>(local_count), 0.0);

  const double *lhs_ptr = rank_ == 0 ? lhs.data() : nullptr;
  const double *rhs_ptr = rank_ == 0 ? rhs.data() : nullptr;

  MPI_Scatterv(lhs_ptr, counts.data(), displs.data(), MPI_DOUBLE, local_lhs_.data(), local_count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
  MPI_Scatterv(rhs_ptr, counts.data(), displs.data(), MPI_DOUBLE, local_rhs_.data(), local_count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  local_sum_ = 0.0;
  result_ = 0.0;
  return true;
}

bool VectorScalarProductMpi::RunImpl() {
  local_sum_ = std::inner_product(local_lhs_.begin(), local_lhs_.end(), local_rhs_.begin(), 0.0);

  return MPI_Reduce(&local_sum_, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
}

bool VectorScalarProductMpi::PostProcessingImpl() {
  MPI_Bcast(&result_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result_;
  return true;
}

}  // namespace vector_scalar_product
