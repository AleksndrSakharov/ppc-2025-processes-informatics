#include "eremin_v_rectangle_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>

#include "eremin_v_rectangle_method/common/include/common.hpp"

namespace eremin_v_rectangle_method {

EreminVRectangleMethodMPI::EreminVRectangleMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVRectangleMethodMPI::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVRectangleMethodMPI::PreProcessingImpl() {
  return true;
}

bool EreminVRectangleMethodMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double lower_bound = 0.0;
  double upper_bound = 0.0;
  int steps = 0;
  double result = 0.0;

  if (rank == 0) {
    auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    steps = std::get<2>(input);
  }
  MPI_Bcast(&lower_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&upper_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const auto in_function = std::get<3>(GetInput());

  double step_size = (upper_bound - lower_bound) / static_cast<double>(steps);
  double local_result = 0.0;

  for (int i = rank; i < steps; i += size) {
    local_result += in_function(lower_bound + ((static_cast<double>(i) + 0.5) * step_size));
  }

  local_result *= step_size;

  MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = result;
  return true;
}

bool EreminVRectangleMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_rectangle_method
