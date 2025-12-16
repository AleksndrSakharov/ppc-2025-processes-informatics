#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lifanov_k_adj_inv_count_restore/common/include/common.hpp"
#include "lifanov_k_adj_inv_count_restore/mpi/include/ops_mpi.hpp"
#include "lifanov_k_adj_inv_count_restore/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lifanov_k_adj_inv_count_restore {

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

class LifanovKRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const ::testing::TestParamInfo<FuncParam> &info) {
    return std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  InType GetTestInputData() final {
    return data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

 private:
  InType data_;
  OutType expected_{};
};

namespace {

std::vector<FuncParam> LoadTestParams() {
  const std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lifanov_k_adj_inv_count_restore, "tests.json");
  const std::string settings_path = PPC_SETTINGS_lifanov_k_adj_inv_count_restore;
  const std::string mpi_suffix =
      ppc::task::GetStringTaskType(LifanovKAdjacentInversionCountMPI::GetStaticTypeOfTask(), settings_path);
  const std::string seq_suffix =
      ppc::task::GetStringTaskType(LifanovKAdjacentInversionCountSEQ::GetStaticTypeOfTask(), settings_path);

  std::ifstream fin(path);
  if (!fin.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  nlohmann::json j;
  fin >> j;

  std::vector<FuncParam> cases;
  cases.reserve(j.size() * 2);

  for (const auto &item : j) {
    TestType tc{item.at("input").get<InType>(), item.at("expected").get<OutType>(), item.at("name").get<std::string>()};

    const std::string &tname = std::get<2>(tc);

    std::string mpi_name = tname;
    mpi_name += '_';
    mpi_name += mpi_suffix;
    cases.emplace_back(ppc::task::TaskGetter<LifanovKAdjacentInversionCountMPI, InType>, std::move(mpi_name), tc);

    std::string seq_name = tname;
    seq_name += '_';
    seq_name += seq_suffix;
    cases.emplace_back(ppc::task::TaskGetter<LifanovKAdjacentInversionCountSEQ, InType>, std::move(seq_name), tc);
  }

  return cases;
}

const std::vector<FuncParam> kFuncParams = LoadTestParams();

TEST_P(LifanovKRunFuncTests, AdjacentInversionCount) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(FunctionalTests, LifanovKRunFuncTests, ::testing::ValuesIn(kFuncParams),
                         LifanovKRunFuncTests::PrintTestParam);

}  // namespace
}  // namespace lifanov_k_adj_inv_count_restore
