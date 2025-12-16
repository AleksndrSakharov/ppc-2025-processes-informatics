#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "baldin_a_gauss_filter/common/include/common.hpp"
#include "baldin_a_gauss_filter/mpi/include/ops_mpi.hpp"
#include "baldin_a_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_gauss_filter {

namespace {

ImageData GetRandomImage(int width, int height, int channels) {
  ImageData data;
  data.width = width;
  data.height = height;
  data.channels = channels;

  size_t size = static_cast<size_t>(width) * height * channels;
  data.pixels.resize(size);

  auto seed = static_cast<unsigned int>(width + height + channels);
  std::mt19937 gen(seed);

  for (size_t i = 0; i < size; i++) {
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

class BaldinAGaussFilterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto [w, h, c] = test_param;
    return "Size_" + std::to_string(w) + "x" + std::to_string(h) + "_Ch" + std::to_string(c);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int width = std::get<0>(params);
    int height = std::get<1>(params);
    int channels = std::get<2>(params);

    input_data_ = GetRandomImage(width, height, channels);
    expected_output_ = CalculateGaussFilter(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_output_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(BaldinAGaussFilterFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(1, 1, 1),     std::make_tuple(1, 1, 3),    std::make_tuple(3, 3, 1),
    std::make_tuple(3, 3, 3),     std::make_tuple(10, 3, 3),

    std::make_tuple(100, 1, 3),   std::make_tuple(1, 100, 3),  std::make_tuple(100, 100, 3),

    std::make_tuple(200, 50, 3),  std::make_tuple(50, 200, 3), std::make_tuple(200, 50, 1),
    std::make_tuple(50, 200, 1),

    std::make_tuple(127, 113, 3),

    std::make_tuple(64, 64, 1),   std::make_tuple(64, 64, 4)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BaldinAGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_baldin_a_gauss_filter),
    ppc::util::AddFuncTask<BaldinAGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_baldin_a_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAGaussFilterFuncTests::PrintFuncTestName<BaldinAGaussFilterFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussFilterTests, BaldinAGaussFilterFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baldin_a_gauss_filter
