#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

namespace {

void ComputeScatterInfo(int total, int size, std::vector<int> &counts, std::vector<int> &displs) {
  const int base = total / size;
  const int rem = total % size;

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
}

void LocalOddEvenPass(std::vector<int> &local, int global_start, int parity) {
  const int n = static_cast<int>(local.size());
  for (int i = 0; i + 1 < n; ++i) {
    const int gidx = global_start + i;
    if ((gidx % 2) == parity) {
      if (local[i] > local[i + 1]) {
        std::swap(local[i], local[i + 1]);
      }
    }
  }
}

void ExchangeBoundary(std::vector<int> &local, const std::vector<int> &counts, int rank, int partner) {
  const int local_n = static_cast<int>(local.size());
  const int partner_n = counts[partner];

  if (local_n == 0 || partner_n == 0) {
    return;
  }

  const bool left_side = (rank < partner);
  int send_value = left_side ? local[local_n - 1] : local[0];
  int recv_value = 0;

  MPI_Sendrecv(&send_value, 1, MPI_INT, partner, 0, &recv_value, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  if (left_side) {
    local[local_n - 1] = std::min(local[local_n - 1], recv_value);
  } else {
    local[0] = std::max(local[0], recv_value);
  }
}

void OddEvenPhase(std::vector<int> &local, const std::vector<int> &counts, const std::vector<int> &displs, int rank,
                  int size, int phase) {
  if (local.empty()) {
    return;
  }

  const int parity = phase % 2;
  const int global_start = displs[rank];

  LocalOddEvenPass(local, global_start, parity);

  const bool even_phase = (phase % 2 == 0);
  const bool even_rank = (rank % 2 == 0);

  int partner = (even_phase == even_rank) ? rank + 1 : rank - 1;

  if (partner >= 0 && partner < size) {
    ExchangeBoundary(local, counts, rank, partner);
  }
}

}  // namespace

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool SizovDBubbleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (rank == 0 ? static_cast<int>(data_.size()) : 0);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n > 0) {
      MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  ComputeScatterInfo(n, size, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv((rank == 0 ? data_.data() : nullptr), counts.data(), displs.data(), MPI_INT, local.data(), local_n,
               MPI_INT, 0, MPI_COMM_WORLD);

  for (int phase = 0; phase < n; ++phase) {
    OddEvenPhase(local, counts, displs, rank, size, phase);
  }

  std::vector<int> result(n);
  MPI_Gatherv(local.data(), local_n, MPI_INT, result.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = result;
  } else {
    GetOutput().assign(n, 0);
  }

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool SizovDBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_bubble_sort
