#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"

namespace chaschin_v_max_for_each_row {

ChaschinVMaxForEachRow::ChaschinVMaxForEachRow(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  this->GetOutput().clear();
}

bool ChaschinVMaxForEachRow::ValidationImpl() {
  const auto &in = GetInput();

  if (in.empty()) {
    return in.empty();
  }

  if (in[0].empty()) {
    return in[0].empty();
  }

  return true;
}

bool ChaschinVMaxForEachRow::PreProcessingImpl() {
  return true;
}

void chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::SendRowsToWorkers(const std::vector<std::vector<float>> &mat,
                                                                            int size) {
  for (int pi = 1; pi < size; ++pi) {
    // Inline ComputeRange
    int nrows = static_cast<int>(mat.size());
    int base = nrows / size;
    int rem = nrows % size;
    int start = (pi * base) + std::min(pi, rem);
    int count = base + (pi < rem ? 1 : 0);

    for (int ii = 0; ii < count; ++ii) {
      const auto &row = mat[start + ii];
      int len = static_cast<int>(row.size());

      MPI_Send(&len, 1, MPI_INT, pi, 100, MPI_COMM_WORLD);
      if (len > 0) {
        MPI_Send(row.data(), len, MPI_FLOAT, pi, 101, MPI_COMM_WORLD);
      }
    }
  }
}

void chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::ReceiveRowsFromRoot(
    std::vector<std::vector<float>> &local_mat) {
  for (auto &row : local_mat) {
    int len = 0;
    MPI_Recv(&len, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    row.resize(len);
    if (len > 0) {
      MPI_Recv(row.data(), len, MPI_FLOAT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

std::vector<std::vector<float>> chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::DistributeRows(
    const std::vector<std::vector<float>> &mat, int rank, int size, const RowRange &range) {
  std::vector<std::vector<float>> local_mat(range.count);

  if (rank == 0) {
    SendRowsToWorkers(mat, size);

    for (int ii = 0; ii < range.count; ++ii) {
      local_mat[ii] = mat[range.start + ii];
    }
  } else {
    ReceiveRowsFromRoot(local_mat);
  }

  return local_mat;
}

std::vector<float> chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::ComputeLocalMax(
    const std::vector<std::vector<float>> &local_mat) {
  std::vector<float> local_out(local_mat.size());
  for (size_t ii = 0; ii < local_mat.size(); ++ii) {
    local_out[ii] = local_mat[ii].empty() ? std::numeric_limits<float>::lowest()
                                          : *std::max_element(local_mat[ii].begin(), local_mat[ii].end());
  }
  return local_out;
}

namespace {
inline void GetRangeForRank(int rank, int total, int world_size, int &start, int &count) {
  int base = total / world_size;
  int rem = total % world_size;
  start = (rank * base) + std::min(rank, rem);
  count = base + (rank < rem ? 1 : 0);
}

inline void RecvRows(int src_rank, std::vector<float> &out, int start, int count) {
  std::vector<float> tmp(count);
  MPI_Recv(tmp.data(), count, MPI_FLOAT, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::ranges::copy(tmp, out.begin() + start);
}
}  // namespace

void chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::GatherResults(std::vector<float> &out,
                                                                        const std::vector<float> &local_out, int rank,
                                                                        int size, const RowRange &range) {
  if (rank != 0) {
    if (!local_out.empty()) {
      MPI_Send(local_out.data(), static_cast<int>(local_out.size()), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
    return;
  }

  for (int i = 0; i < range.count; ++i) {
    out[range.start + i] = local_out[i];
  }

  int total = static_cast<int>(out.size());
  for (int pi = 1; pi < size; ++pi) {
    int start = 0;
    int count = 0;

    GetRangeForRank(pi, total, size, start, count);
    if (count > 0) {
      RecvRows(pi, out, start, count);
    }
  }
}

bool ChaschinVMaxForEachRow::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &mat = GetInput();
  int nrows = (rank == 0) ? static_cast<int>(mat.size()) : 0;
  MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base = nrows / size;
  int rem = nrows % size;
  int start = (rank * base) + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);
  RowRange range{.start = start, .count = count};

  auto local_mat = DistributeRows(mat, rank, size, range);
  auto local_out = ComputeLocalMax(local_mat);

  if (rank == 0) {
    GetOutput().resize(nrows);
  }
  GatherResults(GetOutput(), local_out, rank, size, range);

  auto &out = GetOutput();
  if (rank != 0) {
    out.resize(nrows);
  }

  if (nrows > 0) {
    MPI_Bcast(out.data(), nrows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool ChaschinVMaxForEachRow::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return true;
}
}  // namespace chaschin_v_max_for_each_row
