#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string mpi_opname;
    std::string mpi_typename;
    if (std::get<0>(test_param) == MPI_SUM) {
      mpi_opname = "sum";
    } else if (std::get<0>(test_param) == MPI_MIN) {
      mpi_opname = "min";
    } else {
      throw "unsupported operation";
    }

    if (std::get<1>(test_param) == MPI_INT) {
      mpi_typename = "int";
    } else if (std::get<1>(test_param) == MPI_FLOAT) {
      mpi_typename = "float";
    } else if (std::get<1>(test_param) == MPI_DOUBLE) {
      mpi_typename = "double";
    } else {
      throw "unsupported datatype";
    }

    return mpi_opname + "_" + mpi_typename + "_" + std::to_string(std::get<2>(test_param)) + "_" +
           std::to_string(std::get<3>(test_param));
  }

 protected:
  void SetUp() override {
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized == 0) {
      return;
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    MPI_Op operation = std::get<0>(params);
    MPI_Datatype cur_type = std::get<1>(params);
    size_t vec_size = std::get<2>(params);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::shared_ptr<void> inp_data;

    if (cur_type == MPI_INT) {
      auto *raw_data = new int[vec_size];
      inp_data = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<int *>(p); });
    } else if (cur_type == MPI_FLOAT) {
      auto *raw_data = new float[vec_size];
      inp_data = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<float *>(p); });
    } else if (cur_type == MPI_DOUBLE) {
      auto *raw_data = new double[vec_size];
      inp_data = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<double *>(p); });
    } else {
      throw "unsupported datatype";
    }

    if (operation == MPI_SUM) {
      for (unsigned int i = 0; i < vec_size; i++) {
        if (cur_type == MPI_INT) {
          (static_cast<int *>(inp_data.get()))[i] = static_cast<int>(i) + 3;
        } else if (cur_type == MPI_FLOAT) {
          (static_cast<float *>(inp_data.get()))[i] = (static_cast<float>(i) * 1.1F) + 3.0F;
        } else if (cur_type == MPI_DOUBLE) {
          (static_cast<double *>(inp_data.get()))[i] = (static_cast<double>(i) * 1.1) + 3.0;
        }
      }
    } else if (operation == MPI_MIN) {
      for (unsigned int i = 0; i < vec_size; i++) {
        if (cur_type == MPI_INT) {
          (static_cast<int *>(inp_data.get()))[i] = 10000 - (3 * rank);
        } else if (cur_type == MPI_FLOAT) {
          (static_cast<float *>(inp_data.get()))[i] = 10000.0F - (3.0F * static_cast<float>(rank));
        } else if (cur_type == MPI_DOUBLE) {
          (static_cast<double *>(inp_data.get()))[i] = 10000.0 - (3.0 * static_cast<double>(rank));
        }
      }
    } else {
      throw "unsupported operation";
    }

    input_data_ = std::make_tuple(operation, cur_type, vec_size, inp_data, std::get<3>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<1>(output_data)) {
      return true;
    }

    std::shared_ptr<void> result_ptr = std::get<0>(output_data);
    if (!result_ptr) {
      return false;
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    MPI_Op operation = std::get<0>(params);
    MPI_Datatype data_type = std::get<1>(params);
    size_t vec_size = std::get<2>(params);

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::shared_ptr<void> input_ptr = std::get<3>(input_data_);

    if (operation == MPI_SUM) {
      if (data_type == MPI_INT) {
        return HandleSum(static_cast<int *>(input_ptr.get()), static_cast<int *>(result_ptr.get()), vec_size,
                         world_size);
      }
      if (data_type == MPI_FLOAT) {
        return HandleSum(static_cast<float *>(input_ptr.get()), static_cast<float *>(result_ptr.get()), vec_size,
                         world_size);
      }
      if (data_type == MPI_DOUBLE) {
        return HandleSum(static_cast<double *>(input_ptr.get()), static_cast<double *>(result_ptr.get()), vec_size,
                         world_size);
      }
    }

    if (operation == MPI_MIN) {
      if (data_type == MPI_INT) {
        return HandleMin(static_cast<int *>(result_ptr.get()), vec_size, world_size);
      }
      if (data_type == MPI_FLOAT) {
        return HandleMin(static_cast<float *>(result_ptr.get()), vec_size, world_size);
      }
      if (data_type == MPI_DOUBLE) {
        return HandleMin(static_cast<double *>(result_ptr.get()), vec_size, world_size);
      }
    }

    return false;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  template <typename T>
  static bool CheckEqualArrays(const T *expected, const T *actual, size_t size, double eps) {
    for (size_t i = 0; i < size; i++) {
      if constexpr (std::is_floating_point_v<T>) {
        if (std::fabs(expected[i] - actual[i]) > eps) {
          return false;
        }
      } else {
        if (expected[i] != actual[i]) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename T>
  static bool HandleSum(const T *input, const T *output, size_t size, int world_size) {
    std::vector<T> expected(size);
    T multiplier = static_cast<T>(world_size);

    for (size_t i = 0; i < size; i++) {
      expected[i] = input[i] * multiplier;
    }

    double eps = 0.0;
    if (std::is_same_v<T, float>) {
      eps = 1e-4F;
    } else if (std::is_same_v<T, double>) {
      eps = 1e-9F;
    }

    return CheckEqualArrays(expected.data(), output, size, eps);
  }

  template <typename T>
  static bool HandleMin(const T *output, size_t size, int world_size) {
    std::vector<T> expected(size);
    T value = static_cast<T>(10000 - (3 * (world_size - 1)));

    for (size_t i = 0; i < size; i++) {
      expected[i] = value;
    }

    double eps = 0.0;
    if (std::is_same_v<T, float>) {
      eps = 1e-4F;
    } else if (std::is_same_v<T, double>) {
      eps = 1e-9F;
    }

    return CheckEqualArrays(expected.data(), output, size, eps);
  }
};

namespace {

TEST_P(ZavyalovAReduceFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 36> kTestParam = {
    std::make_tuple(MPI_SUM, MPI_INT, 10U, 0),    std::make_tuple(MPI_SUM, MPI_INT, 10U, 1),
    std::make_tuple(MPI_SUM, MPI_INT, 9U, 0),     std::make_tuple(MPI_SUM, MPI_INT, 9U, 1),
    std::make_tuple(MPI_SUM, MPI_FLOAT, 10U, 0),  std::make_tuple(MPI_SUM, MPI_FLOAT, 10U, 1),
    std::make_tuple(MPI_SUM, MPI_FLOAT, 9U, 0),   std::make_tuple(MPI_SUM, MPI_FLOAT, 9U, 1),
    std::make_tuple(MPI_SUM, MPI_DOUBLE, 10U, 0), std::make_tuple(MPI_SUM, MPI_DOUBLE, 10U, 1),
    std::make_tuple(MPI_SUM, MPI_DOUBLE, 9U, 0),  std::make_tuple(MPI_SUM, MPI_DOUBLE, 9U, 1),
    std::make_tuple(MPI_SUM, MPI_FLOAT, 50U, 0),  std::make_tuple(MPI_SUM, MPI_INT, 50U, 0),
    std::make_tuple(MPI_SUM, MPI_DOUBLE, 50U, 0), std::make_tuple(MPI_SUM, MPI_FLOAT, 50U, 1),
    std::make_tuple(MPI_SUM, MPI_INT, 50U, 1),    std::make_tuple(MPI_SUM, MPI_DOUBLE, 50U, 1),
    std::make_tuple(MPI_MIN, MPI_INT, 10U, 0),    std::make_tuple(MPI_MIN, MPI_INT, 10U, 1),
    std::make_tuple(MPI_MIN, MPI_INT, 9U, 0),     std::make_tuple(MPI_MIN, MPI_INT, 9U, 1),
    std::make_tuple(MPI_MIN, MPI_FLOAT, 10U, 0),  std::make_tuple(MPI_MIN, MPI_FLOAT, 10U, 1),
    std::make_tuple(MPI_MIN, MPI_FLOAT, 9U, 0),   std::make_tuple(MPI_MIN, MPI_FLOAT, 9U, 1),
    std::make_tuple(MPI_MIN, MPI_DOUBLE, 10U, 0), std::make_tuple(MPI_MIN, MPI_DOUBLE, 10U, 1),
    std::make_tuple(MPI_MIN, MPI_DOUBLE, 9U, 0),  std::make_tuple(MPI_MIN, MPI_DOUBLE, 9U, 1),
    std::make_tuple(MPI_MIN, MPI_FLOAT, 50U, 0),  std::make_tuple(MPI_MIN, MPI_INT, 50U, 0),
    std::make_tuple(MPI_MIN, MPI_DOUBLE, 50U, 0), std::make_tuple(MPI_MIN, MPI_FLOAT, 50U, 1),
    std::make_tuple(MPI_MIN, MPI_INT, 50U, 1),    std::make_tuple(MPI_MIN, MPI_DOUBLE, 50U, 1),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ZavyalovAReduceMPI, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce),
                   ppc::util::AddFuncTask<ZavyalovAReduceSEQ, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZavyalovAReduceFuncTests::PrintFuncTestName<ZavyalovAReduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZavyalovAReduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zavyalov_a_reduce
