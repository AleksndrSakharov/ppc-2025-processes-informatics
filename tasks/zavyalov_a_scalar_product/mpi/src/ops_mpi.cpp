#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "zavyalov_a_scalar_product/common/include/common.hpp"

namespace zavyalov_a_scalar_product {

ZavyalovAScalarProductMPI::ZavyalovAScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = 0.0;
}

bool ZavyalovAScalarProductMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  return (!std::get<0>(GetInput()).empty()) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAScalarProductMPI::PreProcessingImpl() {
  return true;
}
bool ZavyalovAScalarProductMPI::RunImpl() {
  const double *left_data = nullptr;
  const double *right_data = nullptr;

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vector_size = 0;

  if (rank == 0) {
    GetOutput() = 0.0;
    const auto &input = GetInput();
    if (!std::get<0>(input).empty()) {  // it does not compile in ubuntu without this line
      left_data = std::get<0>(input).data();
    }
    if (!std::get<1>(input).empty()) {  // it does not compile in ubuntu without this line
      right_data = std::get<1>(input).data();
    }
    vector_size = static_cast<int>(std::get<0>(input).size());
  }

  MPI_Bcast(&vector_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  int blocksize = vector_size / world_size;
  int elements_left = vector_size - (world_size * blocksize);
  int elements_processed = 0;

  for (int i = 0; i < world_size; i++) {
    sendcounts[i] = blocksize + (i < elements_left ? 1 : 0);
    displs[i] = elements_processed;
    elements_processed += sendcounts[i];
  }

  int elements_count = sendcounts[rank];
  std::vector<double> local_left(elements_count);
  std::vector<double> local_right(elements_count);

  MPI_Scatterv(left_data, sendcounts.data(), displs.data(), MPI_DOUBLE, local_left.data(), elements_count, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  MPI_Scatterv(right_data, sendcounts.data(), displs.data(), MPI_DOUBLE, local_right.data(), elements_count, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  double cur_res = 0.0;
  for (int i = 0; i < elements_count; i++) {
    cur_res += local_left[i] * local_right[i];
  }

  double glob_res = 0.0;
  MPI_Allreduce(&cur_res, &glob_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = glob_res;

  return true;
}

bool ZavyalovAScalarProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_scalar_product
