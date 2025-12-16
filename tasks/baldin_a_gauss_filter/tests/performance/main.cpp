#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "baldin_a_gauss_filter/common/include/common.hpp"
#include "baldin_a_gauss_filter/mpi/include/ops_mpi.hpp"
#include "baldin_a_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baldin_a_gauss_filter {

namespace {

ImageData GetRandomImage(int width, int height) {
  ImageData data;
  data.width = width;
  data.height = height;
  data.channels = 3;

  size_t size = static_cast<size_t>(width) * height * 3;
  data.pixels.resize(size);

  auto seed = static_cast<unsigned int>(width + height + 3);
  std::mt19937 gen(seed);

  for (size_t i = 0; i < size; ++i) {
    data.pixels[i] = static_cast<uint8_t>(gen() % 256);
  }

  return data;
}

ImageData CalculateGaussFilter(const ImageData &src) {
  ImageData dst = src;
  int w = src.width;
  int h = src.height;
  int c = src.channels;

  constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      for (int ch = 0; ch < c; ch++) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            int ny = std::clamp(row + dy, 0, h - 1);
            int nx = std::clamp(col + dx, 0, w - 1);

            int pixel_val = src.pixels[((ny * w + nx) * c) + ch];
            int kernel_val = kKernel.at((static_cast<size_t>(dy + 1) * 3) + (dx + 1));
            sum += pixel_val * kernel_val;
          }
        }
        dst.pixels[((row * w + col) * c) + ch] = static_cast<uint8_t>(sum / 16);
      }
    }
  }

  return dst;
}

}  // namespace

class BaldinAGaussFilterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const int width = 3000;
    const int height = 3000;

    input_data_ = GetRandomImage(width, height);
    expected_output_ = CalculateGaussFilter(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BaldinAGaussFilterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BaldinAGaussFilterMPI, BaldinAGaussFilterSEQ>(
    PPC_SETTINGS_baldin_a_gauss_filter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaldinAGaussFilterPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaldinAGaussFilterPerfTests, kGtestValues, kPerfTestName);

}  // namespace baldin_a_gauss_filter
