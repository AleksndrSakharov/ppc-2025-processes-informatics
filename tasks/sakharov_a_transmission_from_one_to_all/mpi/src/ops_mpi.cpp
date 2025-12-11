#include "sakharov_a_transmission_from_one_to_all/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"

namespace sakharov_a_transmission_from_one_to_all {

int My_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int v_rank = (rank - root + size) % size;

  if (v_rank != 0) {
    int v_parent = (v_rank - 1) / 2;
    int real_parent = (v_parent + root) % size;
    MPI_Recv(buffer, count, datatype, real_parent, 0, comm, MPI_STATUS_IGNORE);
  }

  int v_child1 = 2 * v_rank + 1;
  if (v_child1 < size) {
    int real_child1 = (v_child1 + root) % size;
    MPI_Send(buffer, count, datatype, real_child1, 0, comm);
  }

  int v_child2 = 2 * v_rank + 2;
  if (v_child2 < size) {
    int real_child2 = (v_child2 + root) % size;
    MPI_Send(buffer, count, datatype, real_child2, 0, comm);
  }

  return MPI_SUCCESS;
}

SakharovATransmissionFromOneToAllMPI::SakharovATransmissionFromOneToAllMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovATransmissionFromOneToAllMPI::ValidationImpl() {
  int root = std::get<0>(GetInput());
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return root >= 0 && root < world_size;
}

bool SakharovATransmissionFromOneToAllMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SakharovATransmissionFromOneToAllMPI::RunImpl() {
  int root = std::get<0>(GetInput());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // 1. Test MPI_INT
  std::vector<int> data;
  int count = 0;

  if (rank == root) {
    data = std::get<1>(GetInput());
    count = static_cast<int>(data.size());
  }

  My_Bcast(&count, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (rank != root) {
    data.resize(count);
  }

  My_Bcast(data.data(), count, MPI_INT, root, MPI_COMM_WORLD);

  GetOutput() = data;

  // 2. Test MPI_FLOAT
  std::vector<float> float_data;
  int float_count = 100;
  if (rank == root) {
    float_data.resize(float_count);
    for (int i = 0; i < float_count; ++i) {
      float_data[i] = static_cast<float>(i) + 0.5f;
    }
  } else {
    float_data.resize(float_count);
  }

  My_Bcast(float_data.data(), float_count, MPI_FLOAT, root, MPI_COMM_WORLD);

  for (int i = 0; i < float_count; ++i) {
    if (std::abs(float_data[i] - (static_cast<float>(i) + 0.5f)) > 1e-5f) {
      return false;
    }
  }

  // 3. Test MPI_DOUBLE
  std::vector<double> double_data;
  int double_count = 100;
  if (rank == root) {
    double_data.resize(double_count);
    for (int i = 0; i < double_count; ++i) {
      double_data[i] = static_cast<double>(i) + 0.123;
    }
  } else {
    double_data.resize(double_count);
  }

  My_Bcast(double_data.data(), double_count, MPI_DOUBLE, root, MPI_COMM_WORLD);

  for (int i = 0; i < double_count; ++i) {
    if (std::abs(double_data[i] - (static_cast<double>(i) + 0.123)) > 1e-9) {
      return false;
    }
  }

  return true;
}

bool SakharovATransmissionFromOneToAllMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_transmission_from_one_to_all
