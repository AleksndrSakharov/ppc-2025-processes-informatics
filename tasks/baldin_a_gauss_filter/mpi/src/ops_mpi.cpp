#include "baldin_a_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "baldin_a_gauss_filter/common/include/common.hpp"

namespace baldin_a_gauss_filter {

BaldinAGaussFilterMPI::BaldinAGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAGaussFilterMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &im = GetInput();
    bool size_match = (im.pixels.size() == (static_cast<size_t>(im.width) * im.height * im.channels));
    return (im.width > 0 && im.height > 0 && im.channels > 0 && size_match);
  }
  return true;
}

bool BaldinAGaussFilterMPI::PreProcessingImpl() {
  return true;
}

namespace {

void CalculatePartitions(int size, int height, int width, int channels, std::vector<int> &counts,
                         std::vector<int> &displs, std::vector<int> &real_counts) {
  const int rows_per_proc = height / size;
  const int remainder = height % size;

  int current_global_row = 0;
  const int row_size_bytes = width * channels;

  for (int i = 0; i < size; i++) {
    int rows = rows_per_proc + ((i < remainder) ? 1 : 0);
    real_counts[i] = rows;

    int send_start_row = std::max(0, current_global_row - 1);
    int send_end_row = std::min(height - 1, current_global_row + rows);

    int rows_to_send = send_end_row - send_start_row + 1;

    counts[i] = rows_to_send * row_size_bytes;
    displs[i] = send_start_row * row_size_bytes;

    current_global_row += rows;
  }
}

void ComputeHorizontalPass(int rows, int width, int channels, const std::vector<uint8_t> &src,
                           std::vector<uint16_t> &dst) {
  constexpr std::array<int, 3> kKernel = {1, 2, 1};

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < width; col++) {
      for (int ch = 0; ch < channels; ch++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          int n_col = std::clamp(col + k, 0, width - 1);
          sum += src[(((row * width) + n_col) * channels) + ch] * kKernel.at(k + 1);
        }
        dst[(((row * width) + col) * channels) + ch] = static_cast<uint16_t>(sum);
      }
    }
  }
}

void ComputeVerticalPass(int real_rows, int recv_rows, int width, int channels, int row_offset,
                         const std::vector<uint16_t> &src, std::vector<uint8_t> &dst) {
  constexpr std::array<int, 3> kKernel = {1, 2, 1};

  for (int i = 0; i < real_rows; i++) {
    int local_row = row_offset + i;

    for (int col = 0; col < width; col++) {
      for (int ch = 0; ch < channels; ch++) {
        int sum = 0;
        for (int k = -1; k <= 1; k++) {
          int neighbor_row = local_row + k;
          neighbor_row = std::clamp(neighbor_row, 0, recv_rows - 1);

          sum += src[(((neighbor_row * width) + col) * channels) + ch] * kKernel.at(k + 1);
        }
        dst[(((i * width) + col) * channels) + ch] = static_cast<uint8_t>(sum / 16);
      }
    }
  }
}

}  // namespace

bool BaldinAGaussFilterMPI::RunImpl() {
  ImageData &input = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;

  if (rank == 0) {
    width = input.width;
    height = input.height;
    channels = input.channels;
  }
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  std::vector<int> real_counts(size);

  CalculatePartitions(size, height, width, channels, counts, displs, real_counts);

  int my_real_rows = real_counts[rank];
  int my_recv_rows = counts[rank] / (width * channels);

  size_t row_size_bytes = static_cast<size_t>(width) * channels;
  std::vector<uint8_t> local_buffer(static_cast<size_t>(my_recv_rows) * row_size_bytes);
  std::vector<uint16_t> horiz_buffer(static_cast<size_t>(my_recv_rows) * row_size_bytes);
  std::vector<uint8_t> result_buffer(static_cast<size_t>(my_real_rows) * row_size_bytes);

  MPI_Scatterv(rank == 0 ? input.pixels.data() : nullptr, counts.data(), displs.data(), MPI_UINT8_T,
               local_buffer.data(), counts[rank], MPI_UINT8_T, 0, MPI_COMM_WORLD);

  ComputeHorizontalPass(my_recv_rows, width, channels, local_buffer, horiz_buffer);

  int row_offset = (rank == 0) ? 0 : 1;
  ComputeVerticalPass(my_real_rows, my_recv_rows, width, channels, row_offset, horiz_buffer, result_buffer);

  if (rank == 0) {
    GetOutput().width = width;
    GetOutput().height = height;
    GetOutput().channels = channels;
    GetOutput().pixels.resize(static_cast<size_t>(width) * height * channels);
  }

  std::vector<int> recv_counts(size);
  std::vector<int> recv_displs(size);

  if (rank == 0) {
    int current_disp = 0;
    int row_bytes_int = width * channels;
    for (int i = 0; i < size; i++) {
      recv_counts[i] = real_counts[i] * row_bytes_int;
      recv_displs[i] = current_disp;
      current_disp += recv_counts[i];
    }
  }

  MPI_Gatherv(result_buffer.data(), static_cast<int>(result_buffer.size()), MPI_UINT8_T,
              (rank == 0 ? GetOutput().pixels.data() : nullptr), recv_counts.data(), recv_displs.data(), MPI_UINT8_T, 0,
              MPI_COMM_WORLD);

  return true;
}

bool BaldinAGaussFilterMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;
  int channels = 0;

  if (rank == 0) {
    width = GetOutput().width;
    height = GetOutput().height;
    channels = GetOutput().channels;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  size_t total_size = static_cast<size_t>(width) * height * channels;

  if (rank != 0) {
    GetOutput().width = width;
    GetOutput().height = height;
    GetOutput().channels = channels;
    GetOutput().pixels.resize(total_size);
  }

  MPI_Bcast(static_cast<void *>(GetOutput().pixels.data()), static_cast<int>(total_size), MPI_UINT8_T, 0,
            MPI_COMM_WORLD);

  return true;
}

}  // namespace baldin_a_gauss_filter
