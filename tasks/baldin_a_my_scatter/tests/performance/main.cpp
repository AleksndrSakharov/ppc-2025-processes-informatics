#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "baldin_a_my_scatter/common/include/common.hpp"
#include "baldin_a_my_scatter/mpi/include/ops_mpi.hpp"
#include "baldin_a_my_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baldin_a_my_scatter {

class BaldinAMyScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  std::vector<int> send_vec_;
  std::vector<int> recv_vec_;

  const int count_per_proc_ = 10000008;

  bool static IsSeqTest() {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = test_info->name();
    return (test_name.find("seq") != std::string::npos);
  }

  void SetUp() override {
    bool is_seq = IsSeqTest();

    int rank = 0;
    int size = 1;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    int root = 0;
    recv_vec_.resize(count_per_proc_);
    bool i_am_root = is_seq || (rank == root);

    if (i_am_root) {
      size_t total_send_count = is_seq ? count_per_proc_ : (count_per_proc_ * size);

      send_vec_.resize(total_send_count);
      for (size_t i = 0; i < total_send_count; i++) {
        send_vec_[i] = static_cast<int>(i);
      }
    }

    const void *sendbuf_ptr = i_am_root ? send_vec_.data() : nullptr;

    input_data_ = std::make_tuple(sendbuf_ptr, count_per_proc_, MPI_INT, recv_vec_.data(), count_per_proc_, MPI_INT,
                                  root, MPI_COMM_WORLD);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data == nullptr) {
      return false;
    }

    bool is_seq = IsSeqTest();
    int rank = 0;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    int start_value = rank * count_per_proc_;
    const int *actual_data = reinterpret_cast<const int *>(output_data);
    for (int i = 0; i < count_per_proc_; i++) {
      if (actual_data[i] != start_value + i) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BaldinAMyScatterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BaldinAMyScatterMPI, BaldinAMyScatterSEQ>(PPC_SETTINGS_baldin_a_my_scatter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaldinAMyScatterPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaldinAMyScatterPerfTests, kGtestValues, kPerfTestName);

}  // namespace baldin_a_my_scatter
