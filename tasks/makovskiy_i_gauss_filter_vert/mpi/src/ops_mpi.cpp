#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <task/include/task.hpp>
#include <tuple>
#include <vector>

#include "makovskiy_i_gauss_filter_vert/common/include/common.hpp"

namespace makovskiy_i_gauss_filter_vert {

namespace {

int GetPixelValue(int x, int y, int strip_w, int total_h, int rank, const std::vector<int> &all_strip_widths,
                  const std::vector<int> &left_ghost, const std::vector<int> &right_ghost,
                  const std::vector<int> &local_strip) {
  if (x < 0) {
    if (rank > 0 && all_strip_widths.at(static_cast<size_t>(rank) - 1) > 0) {
      return left_ghost.at(static_cast<size_t>(std::clamp(y, 0, total_h - 1)));
    }
    return GetPixel(local_strip, x, y, strip_w, total_h);
  }
  if (x >= strip_w) {
    if (rank < static_cast<int>(all_strip_widths.size()) - 1 &&
        all_strip_widths.at(static_cast<size_t>(rank) + 1) > 0) {
      return right_ghost.at(static_cast<size_t>(std::clamp(y, 0, total_h - 1)));
    }
    return GetPixel(local_strip, x, y, strip_w, total_h);
  }
  return GetPixel(local_strip, x, y, strip_w, total_h);
}

int ApplyKernel(int row, int col, int strip_w, int total_h, int rank, const std::vector<int> &all_strip_widths,
                const std::vector<int> &left_ghost, const std::vector<int> &right_ghost,
                const std::vector<int> &local_strip) {
  const std::array<int, 9> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int sum = 0;

  for (int k_row = -1; k_row <= 1; ++k_row) {
    for (int k_col = -1; k_col <= 1; ++k_col) {
      int current_x = col + k_col;
      int current_y = row + k_row;

      int pixel_val = GetPixelValue(current_x, current_y, strip_w, total_h, rank, all_strip_widths, left_ghost,
                                    right_ghost, local_strip);

      sum += pixel_val * kernel.at((static_cast<size_t>(k_row + 1) * 3) + static_cast<size_t>(k_col + 1));
    }
  }
  return sum;
}

}  // namespace

GaussFilterMPI::GaussFilterMPI(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(ppc::task::TypeOfTask::kMPI);
}

bool GaussFilterMPI::ValidationImpl() {
  const auto &[input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width) * height;
}

bool GaussFilterMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &[_, width, height] = GetInput();
    total_width_ = width;
    total_height_ = height;
  }

  MPI_Bcast(&total_width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int min_strip_width = total_width_ / world_size;
  const int remainder = total_width_ % world_size;
  strip_width_ = min_strip_width + (rank < remainder ? 1 : 0);

  if (strip_width_ > 0) {
    local_strip_.resize(static_cast<size_t>(strip_width_) * total_height_);
  }

  if (rank == 0) {
    GetOutput().resize(static_cast<size_t>(total_width_) * total_height_);
  }

  return true;
}

void GaussFilterMPI::ScatterDataRoot(int world_size) {
  const auto &[input, _w, _h] = GetInput();
  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    int current_strip_width = (total_width_ / world_size) + (i < (total_width_ % world_size) ? 1 : 0);
    if (current_strip_width == 0) {
      continue;
    }

    std::vector<int> strip_to_send(static_cast<size_t>(current_strip_width) * total_height_);
    for (int row = 0; row < total_height_; ++row) {
      for (int col = 0; col < current_strip_width; ++col) {
        strip_to_send.at((static_cast<size_t>(row) * current_strip_width) + col) =
            input.at((static_cast<size_t>(row) * total_width_) + offset + col);
      }
    }

    if (i == 0) {
      local_strip_ = strip_to_send;
    } else {
      MPI_Send(strip_to_send.data(), static_cast<int>(strip_to_send.size()), MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    offset += current_strip_width;
  }
}

void GaussFilterMPI::ScatterDataLeaf() {
  if (strip_width_ > 0) {
    MPI_Recv(local_strip_.data(), static_cast<int>(local_strip_.size()), MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void GaussFilterMPI::ScatterData(int rank, int world_size) {
  if (rank == 0) {
    ScatterDataRoot(world_size);
  } else {
    ScatterDataLeaf();
  }
}

std::vector<int> GaussFilterMPI::ComputeLocal(int rank, int world_size) {
  std::vector<int> local_output(static_cast<size_t>(strip_width_) * total_height_);
  std::vector<int> all_strip_widths(world_size);
  MPI_Allgather(&strip_width_, 1, MPI_INT, all_strip_widths.data(), 1, MPI_INT, MPI_COMM_WORLD);

  if (strip_width_ > 0) {
    std::vector<int> left_ghost(total_height_);
    std::vector<int> right_ghost(total_height_);
    std::array<MPI_Request, 4> requests{};
    int req_count = 0;

    std::vector<int> left_border(total_height_);
    std::vector<int> right_border(total_height_);

    for (int row = 0; row < total_height_; ++row) {
      left_border.at(row) = local_strip_.at(static_cast<size_t>(row) * strip_width_);
      right_border.at(row) = local_strip_.at((static_cast<size_t>(row) * strip_width_) + strip_width_ - 1);
    }

    if (rank > 0 && all_strip_widths.at(static_cast<size_t>(rank) - 1) > 0) {
      MPI_Isend(left_border.data(), total_height_, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests.at(req_count++));
      MPI_Irecv(left_ghost.data(), total_height_, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &requests.at(req_count++));
    }
    if (rank < world_size - 1 && all_strip_widths.at(static_cast<size_t>(rank) + 1) > 0) {
      MPI_Isend(right_border.data(), total_height_, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &requests.at(req_count++));
      MPI_Irecv(right_ghost.data(), total_height_, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &requests.at(req_count++));
    }
    MPI_Waitall(req_count, requests.data(), MPI_STATUSES_IGNORE);

    const int kernel_sum = 16;
    for (int row = 0; row < total_height_; ++row) {
      for (int col = 0; col < strip_width_; ++col) {
        int sum = ApplyKernel(row, col, strip_width_, total_height_, rank, all_strip_widths, left_ghost, right_ghost,
                              local_strip_);
        local_output.at((static_cast<size_t>(row) * strip_width_) + col) = sum / kernel_sum;
      }
    }
  }
  return local_output;
}

void GaussFilterMPI::GatherDataRoot(int world_size, std::vector<int> &final_output,
                                    const std::vector<int> &local_output) const {
  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    int current_strip_width = (total_width_ / world_size) + (i < (total_width_ % world_size) ? 1 : 0);
    if (current_strip_width == 0) {
      continue;
    }

    std::vector<int> received_strip(static_cast<size_t>(current_strip_width) * total_height_);
    if (i == 0) {
      received_strip = local_output;
    } else {
      MPI_Recv(received_strip.data(), static_cast<int>(received_strip.size()), MPI_INT, i, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    for (int row = 0; row < total_height_; ++row) {
      for (int col = 0; col < current_strip_width; ++col) {
        final_output.at((static_cast<size_t>(row) * total_width_) + offset + col) =
            received_strip.at((static_cast<size_t>(row) * current_strip_width) + col);
      }
    }
    offset += current_strip_width;
  }
}

void GaussFilterMPI::GatherDataLeaf(const std::vector<int> &local_output) const {
  if (strip_width_ > 0) {
    MPI_Send(local_output.data(), static_cast<int>(local_output.size()), MPI_INT, 0, 1, MPI_COMM_WORLD);
  }
}

void GaussFilterMPI::GatherData(int rank, int world_size, const std::vector<int> &local_output) {
  if (rank == 0) {
    auto &final_output = GetOutput();
    GatherDataRoot(world_size, final_output, local_output);
  } else {
    GatherDataLeaf(local_output);
  }
}

bool GaussFilterMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  ScatterData(rank, world_size);
  std::vector<int> local_output = ComputeLocal(rank, world_size);
  MPI_Barrier(MPI_COMM_WORLD);
  GatherData(rank, world_size, local_output);

  return true;
}

bool GaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_gauss_filter_vert
