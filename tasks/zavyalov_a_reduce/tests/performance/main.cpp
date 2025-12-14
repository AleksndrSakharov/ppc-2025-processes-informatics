#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>

#include "util/include/perf_test_util.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReducePerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const uint64_t kCount_ = 20000000ULL;
  InType input_data_;
  std::shared_ptr<void> data_;

  void SetUp() override {
    int *raw_data = new int[kCount_];
    data_ = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<int *>(p); });

    int *int_data = static_cast<int *>(data_.get());
    for (uint64_t i = 0; i < kCount_; i++) {
      int_data[i] = 1;
    }

    input_data_ = std::make_tuple(MPI_SUM, MPI_INT, kCount_, data_, 0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<1>(output_data)) {
      return true;
    }

    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized == 0) {
      return true;
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::shared_ptr<void> result_ptr = std::get<0>(output_data);
    if (!result_ptr) {
      return false;
    }

    int *result_data = static_cast<int *>(result_ptr.get());
    bool ok = true;
    for (uint64_t i = 0; i < kCount_; i++) {
      ok &= (result_data[i] == world_size);
    }
    return ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZavyalovAReducePerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZavyalovAReduceMPI, ZavyalovAReduceSEQ>(PPC_SETTINGS_zavyalov_a_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZavyalovAReducePerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZavyalovAReducePerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zavyalov_a_reduce
