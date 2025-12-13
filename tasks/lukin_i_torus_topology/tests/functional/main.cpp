#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "lukin_i_torus_topology/mpi/include/ops_mpi.hpp"
#include "lukin_i_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lukin_i_torus_topology {

class LukinIRunFuncTestsProceses3 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string test_name = test_param;
    return test_name;
  }

 protected:  // тесты написаны исходя из того, что тестирование осуществляется на 2-4 процессах, как и в CI
  void SetUp() override {
    int is_init = 0;
    MPI_Initialized(&is_init);

    if (is_init == 0) {
      return;
    }

    int proc_count = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    if (proc_count == 1) {
      input_data_ = std::make_tuple(0, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
      return;
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (params == "send_myself") {
      SendMyself();
    } else if (params == "send_to_right_neighbour") {
      SendRightNeighbour();
    } else if (params == "send_to_left_neighbour") {
      SendLeftNeighbour();
    } else if (params == "send_to_bottom_neighbour") {
      SendBottomNeighbour(proc_count);
    } else if (params == "send_to_top_neighbour") {
      SendTopNeighbour(proc_count);
    } else if (params == "send_through_right_boundary") {
      SendRightBound(proc_count);
    } else if (params == "send_through_left_boundary") {
      SendLeftBound(proc_count);
    } else if (params == "send_through_bottom_boundary") {
      SendBottomBound(proc_count);
    } else if (params == "send_through_top_boundary") {
      SendTopBound(proc_count);
    }
  }

  void SendMyself() {
    input_data_ = std::make_tuple(0, 0, message_);
    expected_ = std::make_tuple(std::vector<int>{}, message_);
  }

  void SendRightNeighbour() {
    input_data_ = std::make_tuple(0, 1, message_);
    expected_ = std::make_tuple(std::vector<int>{0, 1}, message_);
  }

  void SendLeftNeighbour() {
    input_data_ = std::make_tuple(1, 0, message_);
    expected_ = std::make_tuple(std::vector<int>{1, 0}, message_);
  }

  void SendBottomNeighbour(int proc_count) {
    if (proc_count < 4) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(1, 3, message_);
      expected_ = std::make_tuple(std::vector<int>{1, 3}, message_);
    }
  }

  void SendTopNeighbour(int proc_count) {
    if (proc_count < 4) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(2, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 0}, message_);
    }
  }

  void SendRightBound(int proc_count) {
    if (proc_count == 2) {
      input_data_ = std::make_tuple(1, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{1, 0}, message_);
    } else if (proc_count == 3) {
      input_data_ = std::make_tuple(2, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 0}, message_);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(3, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{3, 2}, message_);
    }
  }

  void SendLeftBound(int proc_count) {
    if (proc_count == 2) {
      input_data_ = std::make_tuple(0, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 1}, message_);
    } else if (proc_count == 3) {
      input_data_ = std::make_tuple(0, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 2}, message_);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(2, 3, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 3}, message_);
    }
  }

  void SendBottomBound(int proc_count) {
    if (proc_count == 2) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (proc_count == 3) {
      input_data_ = std::make_tuple(2, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(2, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 0}, message_);
    }
  }

  void SendTopBound(int proc_count) {
    if (proc_count == 2) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (proc_count == 3) {
      input_data_ = std::make_tuple(2, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (proc_count == 4) {
      input_data_ = std::make_tuple(0, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 2}, message_);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int is_init = 0;
    MPI_Initialized(&is_init);

    if (is_init == 0) {  // защита от threads
      return true;
    }

    std::vector<int> route = std::get<0>(output_data);
    if (route == dummy_) {  // заглушка для seq
      return true;
    }

    return expected_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
  const std::vector<int> message_ = {2, 0, 2, 5};
  const std::vector<int> dummy_ = {2, 0, 2, 6};
};

namespace {

TEST_P(LukinIRunFuncTestsProceses3, ThorTopology) {
  ExecuteTest(GetParam());
}

const std::array<std::string, 9> kTestParam = {"send_myself",
                                               "send_to_right_neighbour",
                                               "send_to_left_neighbour",
                                               "send_to_bottom_neighbour",
                                               "send_to_top_neighbour",
                                               "send_through_right_boundary",
                                               "send_through_left_boundary",
                                               "send_through_bottom_boundary",
                                               "send_through_top_boundary"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LukinIThorTopologyMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_torus_topology),
    ppc::util::AddFuncTask<LukinIThorTopologySEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_torus_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProceses3::PrintFuncTestName<LukinIRunFuncTestsProceses3>;

INSTANTIATE_TEST_SUITE_P(ThorTopologyTests, LukinIRunFuncTestsProceses3, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_torus_topology
