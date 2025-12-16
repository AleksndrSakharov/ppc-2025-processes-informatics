#include "sakharov_a_transmission_from_one_to_all/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"

namespace sakharov_a_transmission_from_one_to_all {

int MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
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

  int v_child1 = (2 * v_rank) + 1;
  if (v_child1 < size) {
    int real_child1 = (v_child1 + root) % size;
    MPI_Send(buffer, count, datatype, real_child1, 0, comm);
  }

  int v_child2 = (2 * v_rank) + 2;
  if (v_child2 < size) {
    int real_child2 = (v_child2 + root) % size;
    MPI_Send(buffer, count, datatype, real_child2, 0, comm);
  }

  return MPI_SUCCESS;
}

namespace {

bool BroadcastInts(int root, int rank, const InType &input, std::vector<int> &data_out) {
  std::vector<int> data;
  int count = 0;

  if (rank == root) {
    data = std::get<1>(input);
    count = static_cast<int>(data.size());
  }

  MyBcast(&count, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (rank != root) {
    data.resize(count);
  }

  MyBcast(data.data(), count, MPI_INT, root, MPI_COMM_WORLD);
  data_out = data;
  return true;
}

bool BroadcastFloats(int root, int rank) {
  std::vector<float> float_data;
  constexpr int kFloatCount = 100;

  if (rank == root) {
    float_data.resize(kFloatCount);
    for (int i = 0; i < kFloatCount; ++i) {
      float_data[i] = static_cast<float>(i) + 0.5F;
    }
  } else {
    float_data.resize(kFloatCount);
  }

  MyBcast(float_data.data(), kFloatCount, MPI_FLOAT, root, MPI_COMM_WORLD);

  for (int i = 0; i < kFloatCount; ++i) {
    if (std::abs(float_data[i] - (static_cast<float>(i) + 0.5F)) > 1e-5F) {
      return false;
    }
  }
  return true;
}

bool BroadcastDoubles(int root, int rank) {
  std::vector<double> double_data;
  constexpr int kDoubleCount = 100;

  if (rank == root) {
    double_data.resize(kDoubleCount);
    for (int i = 0; i < kDoubleCount; ++i) {
      double_data[i] = static_cast<double>(i) + 0.123;
    }
  } else {
    double_data.resize(kDoubleCount);
  }

  MyBcast(double_data.data(), kDoubleCount, MPI_DOUBLE, root, MPI_COMM_WORLD);

  for (int i = 0; i < kDoubleCount; ++i) {
    if (std::abs(double_data[i] - (static_cast<double>(i) + 0.123)) > 1e-9) {
      return false;
    }
  }
  return true;
}

}  // namespace

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
  const int root = std::get<0>(GetInput());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> data;
  if (!BroadcastInts(root, rank, GetInput(), data)) {
    return false;
  }

  if (!BroadcastFloats(root, rank)) {
    return false;
  }

  if (!BroadcastDoubles(root, rank)) {
    return false;
  }

  GetOutput() = data;
  return true;
}

bool SakharovATransmissionFromOneToAllMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_transmission_from_one_to_all
