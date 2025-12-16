#include "romanov_a_integration_rect_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "romanov_a_integration_rect_method/common/include/common.hpp"

namespace romanov_a_integration_rect_method {

RomanovAIntegrationRectMethodMPI::RomanovAIntegrationRectMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool RomanovAIntegrationRectMethodMPI::ValidationImpl() {
  if (!IsEqual(GetOutput(), 0.0)) {
    return false;
  }
  if (std::get<3>(GetInput()) <= 0) {
    return false;
  }
  if (std::get<1>(GetInput()) >= std::get<2>(GetInput())) {
    return false;
  }
  return true;
}

bool RomanovAIntegrationRectMethodMPI::PreProcessingImpl() {
  return true;
}

bool RomanovAIntegrationRectMethodMPI::RunImpl() {
  const auto f = std::get<0>(GetInput());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  double a = 0.0;
  double b = 0.0;
  int n = 0;

  if (rank == 0) {
    a = std::get<1>(GetInput());
    b = std::get<2>(GetInput());
    n = std::get<3>(GetInput());
  }

  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int block_size = (n + num_processes - 1) / num_processes;

  int left_border = rank * block_size;
  int right_border = std::min(n, (rank + 1) * block_size);

  double delta_x = (b - a) / static_cast<double>(n);
  double mid = a + (delta_x * static_cast<double>(left_border)) + (delta_x / 2.0);

  double current_result = 0.0;

  for (int i = left_border; i < right_border; ++i) {
    current_result += f(mid) * delta_x;
    mid += delta_x;
  }

  double result = 0.0;
  MPI_Allreduce(&current_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = result;

  return true;
}

bool RomanovAIntegrationRectMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_integration_rect_method
