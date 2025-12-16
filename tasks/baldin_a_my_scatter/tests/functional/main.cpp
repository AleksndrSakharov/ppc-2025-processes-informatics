#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "baldin_a_my_scatter/common/include/common.hpp"
#include "baldin_a_my_scatter/mpi/include/ops_mpi.hpp"
#include "baldin_a_my_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_my_scatter {

class BaldinAMyScatterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto [count, root, type] = test_param;
    std::string type_str;
    if (type == MPI_INT) {
      type_str = "INT";
    } else if (type == MPI_FLOAT) {
      type_str = "FLOAT";
    } else {
      type_str = "DOUBLE";
    }
    return type_str + "_Count" + std::to_string(count) + "_Root" + std::to_string(root);
  }

  bool static IsSeqTest() {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = test_info->name();
    return (test_name.find("seq") != std::string::npos);
  }

 protected:
  std::vector<int> send_vec_int, recv_vec_int;
  std::vector<float> send_vec_float, recv_vec_float;
  std::vector<double> send_vec_double, recv_vec_double;

  InType input_data;

  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int count = std::get<0>(params);
    int root = std::get<1>(params);
    MPI_Datatype type = std::get<2>(params);

    bool is_seq = IsSeqTest();

    int rank = 0;
    int size = 1;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    int effective_root = root % size;

    const void *sendbuf_ptr = nullptr;
    void *recvbuf_ptr = nullptr;

    bool i_am_root = (rank == effective_root);

    if (type == MPI_INT) {
      recv_vec_int.resize(count);
      recvbuf_ptr = recv_vec_int.data();

      if (i_am_root) {
        size_t total_send_count = is_seq ? count : (count * size);
        send_vec_int.resize(total_send_count);
        for (size_t i = 0; i < total_send_count; i++) {
          send_vec_int[i] = static_cast<int>(i);
        }
        sendbuf_ptr = send_vec_int.data();
      }
    } else if (type == MPI_FLOAT) {
      recv_vec_float.resize(count);
      recvbuf_ptr = recv_vec_float.data();

      if (i_am_root) {
        size_t total_send_count = is_seq ? count : (count * size);
        send_vec_float.resize(total_send_count);
        for (size_t i = 0; i < total_send_count; i++) {
          send_vec_float[i] = static_cast<float>(i);
        }
        sendbuf_ptr = send_vec_float.data();
      }
    } else if (type == MPI_DOUBLE) {
      recv_vec_double.resize(count);
      recvbuf_ptr = recv_vec_double.data();

      if (i_am_root) {
        size_t total_send_count = is_seq ? count : (count * size);
        send_vec_double.resize(total_send_count);
        for (size_t i = 0; i < total_send_count; i++) {
          send_vec_double[i] = static_cast<double>(i);
        }
        sendbuf_ptr = send_vec_double.data();
      }
    }

    input_data = std::make_tuple(sendbuf_ptr, count, type, recvbuf_ptr, count, type, root, MPI_COMM_WORLD);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int count = std::get<0>(params);
    MPI_Datatype type = std::get<2>(params);

    if (output_data == nullptr) {
      return false;
    }

    bool is_seq = IsSeqTest();

    int rank = 0;
    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    int start_value = rank * count;

    if (type == MPI_INT) {
      const auto *actual_data = reinterpret_cast<const int *>(output_data);
      for (int i = 0; i < count; i++) {
        if (actual_data[i] != start_value + i) {
          return false;
        }
      }
    } else if (type == MPI_FLOAT) {
      const auto *actual_data = reinterpret_cast<const float *>(output_data);
      for (int i = 0; i < count; i++) {
        if (std::abs(actual_data[i] - static_cast<float>(start_value + i)) >= 1e-6) {
          return false;
        }
      }
    } else if (type == MPI_DOUBLE) {
      const auto *actual_data = reinterpret_cast<const double *>(output_data);
      for (int i = 0; i < count; i++) {
        if (std::abs(actual_data[i] - static_cast<double>(start_value + i)) >= 1e-10) {
          return false;
        }
      }
    } else {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

namespace {

TEST_P(BaldinAMyScatterFuncTests, MyScatterTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 19> kTestParam = {
    std::make_tuple(1, 0, MPI_INT),

    std::make_tuple(10, 0, MPI_INT),   std::make_tuple(10, 0, MPI_FLOAT),   std::make_tuple(10, 0, MPI_DOUBLE),

    std::make_tuple(10, 1, MPI_INT),   std::make_tuple(10, 1, MPI_FLOAT),   std::make_tuple(10, 1, MPI_DOUBLE),

    std::make_tuple(10, 2, MPI_INT),   std::make_tuple(10, 2, MPI_FLOAT),   std::make_tuple(10, 2, MPI_DOUBLE),

    std::make_tuple(10, 3, MPI_INT),   std::make_tuple(10, 3, MPI_FLOAT),   std::make_tuple(10, 3, MPI_DOUBLE),

    std::make_tuple(17, 0, MPI_INT),   std::make_tuple(123, 0, MPI_INT),    std::make_tuple(7, 1, MPI_DOUBLE),

    std::make_tuple(1000, 0, MPI_INT), std::make_tuple(500, 1, MPI_DOUBLE), std::make_tuple(1500, 2, MPI_FLOAT)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BaldinAMyScatterMPI, InType>(kTestParam, PPC_SETTINGS_baldin_a_my_scatter),
                   ppc::util::AddFuncTask<BaldinAMyScatterSEQ, InType>(kTestParam, PPC_SETTINGS_baldin_a_my_scatter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAMyScatterFuncTests::PrintFuncTestName<BaldinAMyScatterFuncTests>;

INSTANTIATE_TEST_SUITE_P(MyScatterTests, BaldinAMyScatterFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baldin_a_my_scatter
