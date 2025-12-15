#include "smyshlaev_a_gauss_filt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "smyshlaev_a_gauss_filt/seq/include/ops_seq.hpp"

namespace smyshlaev_a_gauss_filt {

namespace {
const std::vector<int> kErnel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
const int kErnelSum = 16;

void FindOptimalGrid(int size, int &grid_rows, int &grid_cols) {
  int best_diff = size;
  grid_rows = 1;
  grid_cols = size;
  for (int rows = 1; rows * rows <= size; ++rows) {
    if (size % rows == 0) {
      int cols = size / rows;
      int diff = std::abs(cols - rows);
      if (diff < best_diff) {
        best_diff = diff;
        grid_rows = rows;
        grid_cols = cols;
      }
    }
  }
}

uint8_t ApplyGaussianFilter(const std::vector<uint8_t> &padded_data, int x, int y, int padded_width, int channels,
                            int channel) {
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int curr_x = x + kx;
      int curr_y = y + ky;
      int pixel = padded_data[(((curr_y * padded_width) + curr_x) * channels) + channel];
      int k_value = kErnel[((ky + 1) * 3) + (kx + 1)];
      sum += pixel * k_value;
    }
  }
  return static_cast<uint8_t>(sum / kErnelSum);
}

}  // namespace

SmyshlaevAGaussFiltMPI::SmyshlaevAGaussFiltMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
}

bool SmyshlaevAGaussFiltMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int error_flag = 0;
  if (rank == 0) {
    const InType &input_img = GetInput();
    if (input_img.width <= 0 || input_img.height <= 0 || input_img.channels <= 0 || input_img.data.empty()) {
      error_flag = 1;
    }
  }
  MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (error_flag == 0);
}

bool SmyshlaevAGaussFiltMPI::PreProcessingImpl() {
  return true;
}

void SmyshlaevAGaussFiltMPI::BroadcastImageDimensions(int &width, int &height, int &channels) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::array<int, 3> dims = {0, 0, 0};
  if (rank == 0) {
    dims[0] = GetInput().width;
    dims[1] = GetInput().height;
    dims[2] = GetInput().channels;
  }
  MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  width = dims[0];
  height = dims[1];
  channels = dims[2];
}

bool SmyshlaevAGaussFiltMPI::RunSequential() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    SmyshlaevAGaussFiltSEQ seq_task(GetInput());
    if (seq_task.Validation()) {
      seq_task.PreProcessing();
      seq_task.Run();
      seq_task.PostProcessing();
      GetOutput() = seq_task.GetOutput();
    }
  }
  return true;
}

void SmyshlaevAGaussFiltMPI::SetupDecomposition(DecompositionInfo &info, int width, int height, int channels) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    FindOptimalGrid(size, info.grid_rows, info.grid_cols);
  }

  MPI_Bcast(&info.grid_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&info.grid_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  info.blocks.resize(size);
  info.sendcounts.resize(size);
  info.displs.resize(size);

  if (rank == 0) {
    int block_height = (height + info.grid_rows - 1) / info.grid_rows;
    int block_width = (width + info.grid_cols - 1) / info.grid_cols;
    for (int pdx = 0; pdx < size; ++pdx) {
      int grid_row = pdx / info.grid_cols;
      int grid_col = pdx % info.grid_cols;
      int start_row = grid_row * block_height;
      int start_col = grid_col * block_width;
      int actual_block_height = std::min(block_height, height - start_row);
      int actual_block_width = std::min(block_width, width - start_col);

      int padded_top = 1;
      int padded_bottom = 1;
      int padded_left = 1;
      int padded_right = 1;

      info.blocks[pdx].start_row = start_row;
      info.blocks[pdx].start_col = start_col;
      info.blocks[pdx].block_height = actual_block_height;
      info.blocks[pdx].block_width = actual_block_width;
      info.blocks[pdx].padded_height = actual_block_height + padded_top + padded_bottom;
      info.blocks[pdx].padded_width = actual_block_width + padded_left + padded_right;
      info.blocks[pdx].count = info.blocks[pdx].padded_height * info.blocks[pdx].padded_width * channels;
      info.sendcounts[pdx] = info.blocks[pdx].count;
    }
    info.displs[0] = 0;
    for (int pdx = 1; pdx < size; ++pdx) {
      info.displs[pdx] = info.displs[pdx - 1] + info.sendcounts[pdx - 1];
    }
  }

  MPI_Bcast(info.sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(info.displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> block_info_buffer(static_cast<size_t>(size) * 6);
  if (rank == 0) {
    for (int pdx = 0; pdx < size; ++pdx) {
      block_info_buffer[(pdx * 6) + 0] = info.blocks[pdx].start_row;
      block_info_buffer[(pdx * 6) + 1] = info.blocks[pdx].start_col;
      block_info_buffer[(pdx * 6) + 2] = info.blocks[pdx].block_height;
      block_info_buffer[(pdx * 6) + 3] = info.blocks[pdx].block_width;
      block_info_buffer[(pdx * 6) + 4] = info.blocks[pdx].padded_height;
      block_info_buffer[(pdx * 6) + 5] = info.blocks[pdx].padded_width;
    }
  }
  MPI_Bcast(block_info_buffer.data(), size * 6, MPI_INT, 0, MPI_COMM_WORLD);

  for (int pdx = 0; pdx < size; ++pdx) {
    info.blocks[pdx].start_row = block_info_buffer[(pdx * 6) + 0];
    info.blocks[pdx].start_col = block_info_buffer[(pdx * 6) + 1];
    info.blocks[pdx].block_height = block_info_buffer[(pdx * 6) + 2];
    info.blocks[pdx].block_width = block_info_buffer[(pdx * 6) + 3];
    info.blocks[pdx].padded_height = block_info_buffer[(pdx * 6) + 4];
    info.blocks[pdx].padded_width = block_info_buffer[(pdx * 6) + 5];
  }
}

std::vector<uint8_t> SmyshlaevAGaussFiltMPI::PrepareScatterBuffer(const DecompositionInfo &info, int width, int height,
                                                                  int channels) {
  int size = static_cast<int>(info.sendcounts.size());
  const auto &input_image = GetInput();

  std::vector<uint8_t> scatter_buffer;
  scatter_buffer.resize(info.displs[size - 1] + info.sendcounts[size - 1]);

  for (int pdx = 0; pdx < size; ++pdx) {
    uint8_t *buffer_ptr = scatter_buffer.data() + info.displs[pdx];
    const auto &block = info.blocks[pdx];
    int src_y_start = block.start_row - 1;
    int src_x_start = block.start_col - 1;
    for (int idy = 0; idy < block.padded_height; ++idy) {
      for (int idx = 0; idx < block.padded_width; ++idx) {
        int global_y = std::clamp(src_y_start + idy, 0, height - 1);
        int global_x = std::clamp(src_x_start + idx, 0, width - 1);
        for (int ch = 0; ch < channels; ++ch) {
          buffer_ptr[(((idy * block.padded_width) + idx) * channels) + ch] =
              input_image.data[(((global_y * width) + global_x) * channels) + ch];
        }
      }
    }
  }
  return scatter_buffer;
}

std::vector<uint8_t> SmyshlaevAGaussFiltMPI::ProcessLocalBlock(const DecompositionInfo &info, int width, int height,
                                                               int channels) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<uint8_t> scatter_buffer;
  if (rank == 0) {
    scatter_buffer = PrepareScatterBuffer(info, width, height, channels);
  }

  std::vector<uint8_t> local_block_data(info.sendcounts[rank]);
  MPI_Scatterv(scatter_buffer.data(), info.sendcounts.data(), info.displs.data(), MPI_UNSIGNED_CHAR,
               local_block_data.data(), info.sendcounts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  const auto &my_block = info.blocks[rank];

  int x_offset = 1;
  int y_offset = 1;

  std::vector<uint8_t> local_output_data(static_cast<size_t>(my_block.block_height) * my_block.block_width * channels);
  for (int idy = 0; idy < my_block.block_height; ++idy) {
    for (int idx = 0; idx < my_block.block_width; ++idx) {
      for (int ch = 0; ch < channels; ++ch) {
        local_output_data[((idy * my_block.block_width + idx) * channels) + ch] =
            ApplyGaussianFilter(local_block_data, idx + x_offset, idy + y_offset, my_block.padded_width, channels, ch);
      }
    }
  }
  return local_output_data;
}

void SmyshlaevAGaussFiltMPI::CollectResult(const std::vector<uint8_t> &local_result, const DecompositionInfo &info,
                                           int width, int height, int channels) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> recvcounts(size);
  std::vector<int> recv_displs(size, 0);
  std::vector<uint8_t> gathered_data;

  if (rank == 0) {
    for (int pdx = 0; pdx < size; ++pdx) {
      recvcounts[pdx] = info.blocks[pdx].block_height * info.blocks[pdx].block_width * channels;
    }
    recv_displs[0] = 0;
    for (int pdx = 1; pdx < size; ++pdx) {
      recv_displs[pdx] = recv_displs[pdx - 1] + recvcounts[pdx - 1];
    }
    gathered_data.resize(static_cast<size_t>(width) * height * channels);
  }

  MPI_Gatherv(local_result.data(), static_cast<int>(local_result.size()), MPI_UNSIGNED_CHAR, gathered_data.data(),
              recvcounts.data(), recv_displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    OutType &output_image = GetOutput();
    output_image.width = width;
    output_image.height = height;
    output_image.channels = channels;
    output_image.data.resize(static_cast<size_t>(width) * height * channels);
    for (int pdx = 0; pdx < size; ++pdx) {
      const uint8_t *src_ptr = gathered_data.data() + recv_displs[pdx];
      const auto &block = info.blocks[pdx];
      for (int idy = 0; idy < block.block_height; ++idy) {
        int global_y = block.start_row + idy;
        uint8_t *dst_ptr = &output_image.data[(static_cast<size_t>(global_y) * width + block.start_col) * channels];
        std::copy_n(src_ptr + (static_cast<size_t>(idy) * block.block_width * channels), block.block_width * channels,
                    dst_ptr);
      }
    }
  }
}

bool SmyshlaevAGaussFiltMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int img_width = 0;
  int img_height = 0;
  int img_channels = 0;
  BroadcastImageDimensions(img_width, img_height, img_channels);

  int grid_rows = 0;
  int grid_cols = 0;
  FindOptimalGrid(size, grid_rows, grid_cols);

  if (img_height < grid_rows || img_width < grid_cols) {
    RunSequential();
    return true;
  }

  DecompositionInfo decomp_info;
  SetupDecomposition(decomp_info, img_width, img_height, img_channels);

  std::vector<uint8_t> local_result = ProcessLocalBlock(decomp_info, img_width, img_height, img_channels);

  CollectResult(local_result, decomp_info, img_width, img_height, img_channels);

  return true;
}

bool SmyshlaevAGaussFiltMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto &output = GetOutput();
  std::array<int, 3> dims = {0, 0, 0};

  if (rank == 0) {
    dims[0] = output.width;
    dims[1] = output.height;
    dims[2] = output.channels;
  }
  MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    output.width = dims[0];
    output.height = dims[1];
    output.channels = dims[2];
    if (dims[0] > 0 && dims[1] > 0 && dims[2] > 0) {
      output.data.resize(static_cast<size_t>(dims[0]) * dims[1] * dims[2]);
    } else {
      output.data.clear();
    }
  }

  if (!output.data.empty()) {
    MPI_Bcast(output.data.data(), static_cast<int>(output.data.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace smyshlaev_a_gauss_filt
