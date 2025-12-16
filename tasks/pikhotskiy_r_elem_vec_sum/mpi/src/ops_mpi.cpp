#include "pikhotskiy_r_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <utility>
#include <vector>

#include "pikhotskiy_r_elem_vec_sum/common/include/common.hpp"

namespace pikhotskiy_r_elem_vec_sum {

PikhotskiyRElemVecSumMPI::PikhotskiyRElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PikhotskiyRElemVecSumMPI::ValidationImpl() {
  const auto &input_data = GetInput();
  bool output_check = (GetOutput() == 0);
  bool size_check = std::cmp_equal(std::get<1>(input_data).size(), std::get<0>(input_data));
  return output_check && size_check;
}

bool PikhotskiyRElemVecSumMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool PikhotskiyRElemVecSumMPI::RunImpl() {
  int my_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_elements = 0;
  if (my_rank == 0) {
    total_elements = std::get<0>(GetInput());
  }

  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_elements == 0) {
    GetOutput() = 0LL;
    return true;
  }

  // Calculate distribution of work
  std::vector<int> counts_per_process(world_size);
  std::vector<int> offsets(world_size);

  int base_count = total_elements / world_size;
  int extra_elements = total_elements % world_size;

  for (int proc = 0; proc < world_size; ++proc) {
    counts_per_process[proc] = base_count + (proc < extra_elements ? 1 : 0);
    offsets[proc] = (proc == 0) ? 0 : offsets[proc - 1] + counts_per_process[proc - 1];
  }

  int my_count = counts_per_process[my_rank];
  std::vector<int> my_data(my_count);

  int *send_buffer = nullptr;
  if (my_rank == 0) {
    send_buffer = const_cast<int *>(std::get<1>(GetInput()).data());
  }

  MPI_Scatterv(send_buffer, counts_per_process.data(), offsets.data(), MPI_INT, my_data.data(), my_count, MPI_INT, 0,
               MPI_COMM_WORLD);

  // Compute local sum
  OutType my_sum = 0LL;
  for (int i = 0; i < my_count; ++i) {
    my_sum += static_cast<OutType>(my_data[i]);
  }

  // Combine all partial sums
  OutType global_result = 0LL;
  MPI_Reduce(&my_sum, &global_result, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // Share result with all processes
  MPI_Bcast(&global_result, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;

  // Synchronize before finishing
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool PikhotskiyRElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_elem_vec_sum
