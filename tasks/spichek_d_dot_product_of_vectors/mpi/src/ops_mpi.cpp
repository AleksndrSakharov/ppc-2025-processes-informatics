#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsMPI::SpichekDDotProductOfVectorsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsMPI::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return vector1.size() == vector2.size();
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[v1, v2] = GetInput();
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(v1.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0 || (rank == 0 && static_cast<size_t>(n) != v2.size())) {
    GetOutput() = 0;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  int base = n / size;
  int rem = n % size;

  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    // [Исправлено] Добавлены скобки для явного приоритета операций
    displs[i] = (i * base) + std::min(i, rem);
  }

  int local_count = counts[rank];

  std::vector<int> lv1(local_count);
  std::vector<int> lv2(local_count);

  MPI_Scatterv(rank == 0 ? v1.data() : nullptr, counts.data(), displs.data(), MPI_INT, lv1.data(), local_count, MPI_INT,
               0, MPI_COMM_WORLD);
  MPI_Scatterv(rank == 0 ? v2.data() : nullptr, counts.data(), displs.data(), MPI_INT, lv2.data(), local_count, MPI_INT,
               0, MPI_COMM_WORLD);

  int64_t local_dot = 0;
  for (int i = 0; i < local_count; ++i) {
    local_dot += static_cast<int64_t>(lv1[i]) * lv2[i];
  }

  int64_t global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot);
  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
